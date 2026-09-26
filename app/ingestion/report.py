"""Ingestion report: the auditable outcome of a run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RowIssue:
    row_number: int
    company_name: str | None
    kind: str  # "validation_error" | "db_error" | "geocode_miss" | "needs_review"
    message: str


MAX_ISSUES_STORED = 500  # keep the JSONB report bounded


@dataclass(slots=True)
class IngestionReport:
    file_name: str
    rows_total: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    needs_review: int = 0
    failed: int = 0
    geocoded: int = 0
    geocode_missing: int = 0
    tenancies_upserted: int = 0
    unknown_columns: list[str] = field(default_factory=list)
    issues: list[RowIssue] = field(default_factory=list)
    resumed_from_row: int = 0

    def add_issue(self, issue: RowIssue) -> None:
        if len(self.issues) < MAX_ISSUES_STORED:
            self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status(self) -> str:
        if self.failed == 0:
            return "succeeded"
        return "failed" if self.failed >= self.rows_total > 0 else "partial"