"""Office-demand signal engine tables: definitions (configurable weights), atomic signals, scores."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.enums import Confidence, EvidenceKind, check_in
from app.models.mixins import ProvenanceMixin, TimestampMixin, UUIDPrimaryKeyMixin


class SignalDefinition(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Configurable signal type. Weights are DATA, not code: tunable without a deploy.

    `rationale` forces every weight to be justified (no unexplained business claims).
    """

    __tablename__ = "signal_definitions"
    __table_args__ = (
        CheckConstraint("weight BETWEEN 0 AND 1", name="weight_range"),
        CheckConstraint("half_life_days > 0", name="half_life_positive"),
        UniqueConstraint("code", name="uq_signal_definitions_code"),
    )

    code: Mapped[str] = mapped_column(String(60), nullable=False)
    label: Mapped[str] = mapped_column(String(150), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    half_life_days: Mapped[int] = mapped_column(nullable=False, default=90, server_default="90")
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True, server_default="true")


class Signal(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """One atomic, sourced piece of evidence about a company.

    `kind` separates verifiable FACT ("25 open Bangalore roles") from interpretive SIGNAL.
    `evidence_text` + `evidence_url` let every claim in an answer be traced and checked.
    """

    __tablename__ = "signals"
    __table_args__ = (
        CheckConstraint(check_in("kind", EvidenceKind), name="kind_valid"),
        CheckConstraint(check_in("confidence", Confidence), name="confidence_valid"),
        Index("ix_signals_company_detected", "company_id", "detected_on"),
        Index("ix_signals_code", "signal_code"),
        # Idempotent recomputation: same company+code+evidence hash is never duplicated.
        UniqueConstraint("company_id", "signal_code", "evidence_hash", name="uq_signals_dedupe"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    signal_code: Mapped[str] = mapped_column(
        String(60), ForeignKey("signal_definitions.code", ondelete="RESTRICT"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(10), nullable=False, default=EvidenceKind.SIGNAL)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_url: Mapped[str | None] = mapped_column(Text)
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_on: Mapped[date] = mapped_column(Date, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)  # e.g. number of roles, % growth
    confidence: Mapped[str] = mapped_column(String(10), nullable=False, default=Confidence.MEDIUM)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class CompanySignalScore(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Reproducible score with the full breakdown stored. No black box: `breakdown` lists every
    contributing signal, its weight, decay factor, and contribution. `config_version` pins the weights used."""

    __tablename__ = "company_signal_scores"
    __table_args__ = (
        CheckConstraint("score BETWEEN 0 AND 100", name="score_range"),
        CheckConstraint(check_in("confidence", Confidence), name="confidence_valid"),
        Index("ix_scores_company_computed", "company_id", "computed_at"),
        Index("ix_scores_score", "score"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_count: Mapped[int] = mapped_column(nullable=False, default=0)
    breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    config_version: Mapped[str] = mapped_column(String(40), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)