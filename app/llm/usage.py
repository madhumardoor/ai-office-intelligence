"""Persist per-call LLM usage.

Only the prompt hash is stored.
Prompt text is never persisted.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.logging import request_id_ctx
from app.llm.types import CallMeta
from app.models import LLMLog


logger = logging.getLogger(__name__)


class UsageRecorder:
    """Records LLM usage without storing prompt contents."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None,
        enabled: bool = True,
    ) -> None:
        self._sf = session_factory
        self._enabled = enabled and session_factory is not None

        self.session_total_cost_usd = 0.0
        self.session_calls = 0

    async def record(self, meta: CallMeta) -> None:
        """Record usage metrics.

        Database logging is best-effort and never breaks the LLM request.
        """
        self.session_calls += 1

        if meta.estimated_cost_usd is not None:
            self.session_total_cost_usd += meta.estimated_cost_usd

        logger.info(
            "llm call",
            extra={
                "purpose": meta.purpose,
                "tier": meta.tier.value,
                "provider": meta.provider,
                "model": meta.model,
                "cache_hit": meta.cache_hit,
                "latency_ms": round(meta.latency_ms, 1),
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
                "estimated_cost_usd": meta.estimated_cost_usd,
            },
        )

        # Cache hits do not represent provider spend and should not
        # create misleading DB usage rows.
        if not self._enabled or meta.cache_hit:
            return

        try:
            assert self._sf is not None

            async with self._sf() as session, session.begin():
                session.add(
                    LLMLog(
                        request_id=request_id_ctx.get(),
                        provider=meta.provider,
                        model=meta.model,
                        purpose=meta.purpose,
                        prompt_hash=meta.prompt_hash,
                        input_tokens=meta.input_tokens,
                        output_tokens=meta.output_tokens,
                        latency_ms=meta.latency_ms,
                        estimated_cost_usd=meta.estimated_cost_usd,
                        error=meta.error,
                    )
                )

        except Exception as exc:  # noqa: BLE001
            # Observability must never break the request path.
            logger.error(
                "failed to persist llm log",
                extra={"error": type(exc).__name__},
            )