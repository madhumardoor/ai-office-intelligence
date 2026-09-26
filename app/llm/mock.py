"""Deterministic mock provider for tests and offline development.

This module never imports LangChain.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from app.llm.types import LLMResult, StructuredResult, Tier, Usage


class MockLLM:
    """Scripted LLM implementation used by tests.

    Records every call and can inject transient failures.
    """

    provider = "mock"

    def __init__(
        self,
        text_fn: Callable[[str, str], str] | None = None,
        structured_fn: Callable[[str, str, type[BaseModel]], Any] | None = None,
        fast_model: str = "mock-fast",
        strong_model: str = "mock-strong",
        fail_times: int = 0,
        exc_factory: Callable[[], Exception] | None = None,
    ) -> None:
        if fail_times < 0:
            raise ValueError("fail_times must be >= 0")

        self._text_fn = text_fn or (lambda _system, _user: "mock answer")
        self._structured_fn = structured_fn

        self._models = {
            Tier.FAST: fast_model,
            Tier.STRONG: strong_model,
        }

        self.calls: list[dict[str, Any]] = []
        self._fail_left = fail_times
        self._exc_factory = exc_factory

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

    def _maybe_fail(self) -> None:
        if self._fail_left <= 0:
            return

        self._fail_left -= 1

        from app.llm.types import LLMTransientError

        raise (
            self._exc_factory()
            if self._exc_factory is not None
            else LLMTransientError("injected failure")
        )

    async def text(
        self,
        tier: Tier,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        self.calls.append(
            {
                "kind": "text",
                "tier": tier,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        self._maybe_fail()

        output = self._text_fn(system, user)

        return LLMResult(
            text=output,
            usage=Usage(
                input_tokens=max(1, len(system + user) // 4),
                output_tokens=max(1, len(output) // 4),
            ),
        )

    async def structured(
        self,
        tier: Tier,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        max_tokens: int,
        temperature: float,
    ) -> StructuredResult[Any]:
        self.calls.append(
            {
                "kind": "structured",
                "tier": tier,
                "system": system,
                "user": user,
                "schema": schema.__name__,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        self._maybe_fail()

        if self._structured_fn is None:
            raise AssertionError(
                "MockLLM.structured_fn not provided for this test"
            )

        raw = self._structured_fn(system, user, schema)

        value = (
            raw
            if isinstance(raw, BaseModel)
            else schema.model_validate(raw)
        )

        return StructuredResult(
            value=value,
            usage=Usage(
                input_tokens=max(1, len(system + user) // 4),
                output_tokens=20,
            ),
        )