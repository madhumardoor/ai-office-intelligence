"""Enumerations stored as text + CHECK constraints (easier to migrate than PG ENUM types)."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    WEB = "web"
    API = "api"
    DOCUMENT = "document"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"


class EvidenceKind(StrEnum):
    """Distinguishes verifiable facts from interpretive signals (core product rule)."""

    FACT = "fact"
    SIGNAL = "signal"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MatchStatus(StrEnum):
    AUTO_MERGED = "auto_merged"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"


class IngestionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class DocumentType(StrEnum):
    NEWS = "news"
    COMPANY_PROFILE = "company_profile"
    CAREERS_PAGE = "careers_page"
    FUNDING_ANNOUNCEMENT = "funding_announcement"
    INTERNAL_DOC = "internal_doc"
    REPORT = "report"
    OTHER = "other"


def check_in(column: str, enum: type[StrEnum]) -> str:
    """Build the SQL for a CHECK (col IN (...)) constraint from an enum."""
    values = ", ".join(f"'{e.value}'" for e in enum)
    return f"{column} IN ({values})"