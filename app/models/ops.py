"""Operational tables: search/LLM logs and web-search cache."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class SearchLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-request retrieval log (latency, tools chosen, retrieved IDs) for observability and eval."""

    __tablename__ = "search_logs"
    __table_args__ = (Index("ix_search_logs_created", "created_at"),)

    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    tools_used: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    retrieved_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float)
    result_count: Mapped[int | None] = mapped_column(Integer)


class LLMLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-LLM-call log: model, tokens, latency, estimated cost. Prompts are NOT stored by default
    (privacy); `prompt_hash` allows cache/debug correlation without retaining content."""

    __tablename__ = "llm_logs"

    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    purpose: Mapped[str] = mapped_column(String(40), nullable=False)  # routing, rag_answer, extraction...
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)


class WebSearchCache(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Cache of web-search results (cost control + reproducibility)."""

    __tablename__ = "web_search_cache"
    __table_args__ = (
        UniqueConstraint("provider", "query_hash", name="uq_web_search_cache_query"),
        Index("ix_web_search_cache_expires", "expires_at"),
    )

    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="SET NULL")
    )
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

class GeocodeCache(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Cache of geocoding lookups, including misses, to respect provider usage policy."""

    __tablename__ = "geocode_cache"
    __table_args__ = (UniqueConstraint("query_hash", name="uq_geocode_cache_query_hash"),)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    found: Mapped[bool] = mapped_column(nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(30), nullable=False, default="nominatim", server_default="nominatim")