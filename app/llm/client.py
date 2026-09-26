"""Provider-independent LLM client.

Composition:
    tiering
    -> cache
    -> resilience
    -> provider
    -> structured-output repair
    -> usage recording

Prompt text is never logged; only a SHA-256 hash is recorded.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel

from app.llm.cache import LRUCache, make_key
from app.llm.pricing import PriceTable
from app.llm.resilience import CircuitBreaker, call_with_resilience
from app.llm.types import (
    CallMeta,
    LLMOutputError,
    ProviderAdapter,
    Tier,
    Usage,
)
from app.llm.usage import UsageRecorder


logger = logging.getLogger(__name__)


_REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be parsed into the required schema. "
    "Reply again with ONLY data that satisfies the schema exactly. "
    "Do not add commentary."
)


class LLMClient:
    """The only LLM object the rest of the application needs."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        *,
        prices: PriceTable | None = None,
        recorder: UsageRecorder | None = None,
        cache: LRUCache | None = None,
        timeout_s: float = 45.0,
        max_retries: int = 3,
        max_output_tokens: int = 1500,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._adapter = adapter
        self._prices = prices or PriceTable()
        self._recorder = recorder or UsageRecorder(None, enabled=False)
        self._cache = cache

        self._timeout = timeout_s
        self._retries = max_retries
        self._max_output_tokens = max_output_tokens

        self._breaker = breaker or CircuitBreaker()
        self._sleep = sleep

    @property
    def provider(self) -> str:
        return self._adapter.provider

    def model_for(self, tier: Tier) -> str:
        return self._adapter.model_for(tier)

    @staticmethod
    def _hash(system: str, user: str) -> str:
        """Hash prompt content without retaining the actual prompt."""
        return hashlib.sha256(
            f"{system}\x1f{user}".encode("utf-8")
        ).hexdigest()

    async def _record_meta(
        self,
        *,
        purpose: str,
        tier: Tier,
        prompt_hash: str,
        started_at: float,
        usage: Usage,
        cache_hit: bool,
        error: str | None = None,
    ) -> None:
        model = self._adapter.model_for(tier)

        await self._recorder.record(
            CallMeta(
                purpose=purpose,
                tier=tier,
                provider=self._adapter.provider,
                model=model,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                estimated_cost_usd=(
                    None
                    if cache_hit
                    else self._prices.estimate(
                        model,
                        usage.input_tokens,
                        usage.output_tokens,
                    )
                ),
                cache_hit=cache_hit,
                prompt_hash=prompt_hash,
                error=error,
            )
        )

    async def generate_text(
        self,
        *,
        purpose: str,
        tier: Tier,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        """Generate plain text with bounded resilience."""

        prompt_hash = self._hash(system, user)
        started_at = time.perf_counter()

        key = (
            make_key(
                "text",
                self._adapter.provider,
                self._adapter.model_for(tier),
                prompt_hash,
                max_output_tokens,
            )
            if self._cache is not None and temperature == 0.0
            else None
        )

        if key is not None:
            cached = self._cache.get(key)

            if cached is not None:
                await self._record_meta(
                    purpose=purpose,
                    tier=tier,
                    prompt_hash=prompt_hash,
                    started_at=started_at,
                    usage=Usage(),
                    cache_hit=True,
                )
                return str(cached)

        try:
            call_kwargs = {
                "max_tokens": max_output_tokens or self._max_output_tokens,
                "temperature": temperature,
            }

            async def provider_call():
                return await self._adapter.text(
                    tier,
                    system,
                    user,
                    **call_kwargs,
                )

            result = await call_with_resilience(
                provider_call,
                breaker=self._breaker,
                timeout_s=self._timeout,
                max_retries=self._retries,
                **({"sleep": self._sleep} if self._sleep else {}),
            )

        except Exception as exc:
            await self._record_meta(
                purpose=purpose,
                tier=tier,
                prompt_hash=prompt_hash,
                started_at=started_at,
                usage=Usage(),
                cache_hit=False,
                error=type(exc).__name__,
            )
            raise

        if key is not None:
            self._cache.put(key, result.text)

        await self._record_meta(
            purpose=purpose,
            tier=tier,
            prompt_hash=prompt_hash,
            started_at=started_at,
            usage=result.usage,
            cache_hit=False,
        )

        return result.text

    async def generate_structured(
        self,
        *,
        purpose: str,
        tier: Tier,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> BaseModel:
        """Generate structured data with exactly one repair attempt."""

        prompt_hash = self._hash(system, user)
        started_at = time.perf_counter()

        key = (
            make_key(
                "struct",
                self._adapter.provider,
                self._adapter.model_for(tier),
                prompt_hash,
                schema.__name__,
                max_output_tokens,
            )
            if self._cache is not None and temperature == 0.0
            else None
        )

        if key is not None:
            cached = self._cache.get(key)

            if cached is not None:
                await self._record_meta(
                    purpose=purpose,
                    tier=tier,
                    prompt_hash=prompt_hash,
                    started_at=started_at,
                    usage=Usage(),
                    cache_hit=True,
                )
                return schema.model_validate(cached)

        total_input = 0
        total_output = 0
        last_error: Exception | None = None

        for attempt in range(2):
            current_user = (
                user
                if attempt == 0
                else user + _REPAIR_SUFFIX
            )

            try:

                async def provider_call():
                    return await self._adapter.structured(
                        tier,
                        system,
                        current_user,
                        schema,
                        max_tokens=max_output_tokens or self._max_output_tokens,
                        temperature=temperature,
                    )

                result = await call_with_resilience(
                    provider_call,
                    breaker=self._breaker,
                    timeout_s=self._timeout,
                    max_retries=self._retries,
                    **({"sleep": self._sleep} if self._sleep else {}),
                )

            except LLMOutputError as exc:
                # Structured parsing failure is handled by our
                # explicit one-time repair path.
                last_error = exc
                continue

            except Exception as exc:
                await self._record_meta(
                    purpose=purpose,
                    tier=tier,
                    prompt_hash=prompt_hash,
                    started_at=started_at,
                    usage=Usage(
                        total_input or None,
                        total_output or None,
                    ),
                    cache_hit=False,
                    error=type(exc).__name__,
                )
                raise

            total_input += result.usage.input_tokens or 0
            total_output += result.usage.output_tokens or 0

            if key is not None:
                self._cache.put(
                    key,
                    result.value.model_dump(mode="json"),
                )

            await self._record_meta(
                purpose=purpose,
                tier=tier,
                prompt_hash=prompt_hash,
                started_at=started_at,
                usage=Usage(
                    total_input or None,
                    total_output or None,
                ),
                cache_hit=False,
            )

            return result.value

        await self._record_meta(
            purpose=purpose,
            tier=tier,
            prompt_hash=prompt_hash,
            started_at=started_at,
            usage=Usage(
                total_input or None,
                total_output or None,
            ),
            cache_hit=False,
            error="LLMOutputError",
        )

        raise LLMOutputError(
            f"structured output failed after repair: {last_error}"
        )