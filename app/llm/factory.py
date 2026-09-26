"""Production LLM client factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.config import Settings
from app.llm.cache import LRUCache
from app.llm.client import LLMClient
from app.llm.pricing import PriceTable
from app.llm.usage import UsageRecorder


def create_llm_client(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> LLMClient:
    """Build the production LLM client."""

    # Lazy import keeps LangChain out of unrelated test paths.
    from app.llm.providers import LangChainAdapter

    return LLMClient(
        LangChainAdapter(settings),
        prices=PriceTable.from_json(
            settings.llm_pricing_json
        ),
        recorder=UsageRecorder(
            session_factory,
            enabled=settings.llm_log_to_db,
        ),
        cache=(
            LRUCache(settings.llm_cache_max_entries)
            if settings.llm_cache_enabled
            else None
        ),
        timeout_s=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_output_tokens=settings.llm_max_output_tokens,
    )