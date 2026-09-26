"""Source registry and ingestion run tracking (provenance + checkpointing)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.enums import IngestionStatus, SourceType, check_in
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Source(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Registry of every data origin (a file, URL, API, or synthetic generator).

    Why it exists: the LLM must cite sources, and every fact row points here.
    """

    __tablename__ = "sources"
    __table_args__ = (
        CheckConstraint(check_in("source_type", SourceType), name="source_type_valid"),
        CheckConstraint("reliability BETWEEN 0 AND 1", name="reliability_range"),
        UniqueConstraint("source_type", "name", name="uq_sources_type_name"),
    )

    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    # Authority weighting used by retrieval/scoring (e.g. company filing > blog).
    reliability: Mapped[float] = mapped_column(nullable=False, default=0.5, server_default="0.5")
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class IngestionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One execution of an ingestion/refresh job.

    Why it exists: checkpointing + resumability + the ingestion report. `checkpoint`
    stores the last successfully processed position so retries never lose or duplicate data.
    """

    __tablename__ = "ingestion_runs"
    __table_args__ = (CheckConstraint(check_in("status", IngestionStatus), name="status_valid"),)

    job_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=IngestionStatus.RUNNING)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    error_summary: Mapped[str | None] = mapped_column(Text)
    report: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    source_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))