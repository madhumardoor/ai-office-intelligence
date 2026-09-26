"""Filters used by hybrid document retrieval."""

from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


# Keep this list deliberately explicit.
# Retrieval should reject unknown document types instead of accepting
# arbitrary strings that could represent unsupported document categories.
ALLOWED_DOCUMENT_TYPES = frozenset(
    {
        "news",
        "company",
        "property",
        "coworking",
        "document",
        "report",
        "website",
        "research",
    }
)


class RetrievalFilters(BaseModel):
    """Optional filters applied to document retrieval."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_types: tuple[str, ...] = ()
    company_ids: tuple[UUID, ...] = ()
    published_after: Optional[date] = None
    published_before: Optional[date] = None

    @field_validator("document_types")
    @classmethod
    def validate_document_types(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        invalid = [
            document_type
            for document_type in value
            if document_type not in ALLOWED_DOCUMENT_TYPES
        ]

        if invalid:
            raise ValueError(
                f"Unsupported document type(s): {', '.join(invalid)}"
            )

        return value

    def to_sql(self, alias: str | None = None) -> tuple[str, dict]:
        """Build parameterised SQL predicates.

        Returns
        -------
        tuple[str, dict]
            SQL fragment and named parameters.

        No user-provided values are interpolated into the SQL string.
        """

        prefix = f"{alias}." if alias else ""

        conditions: list[str] = []
        params: dict = {}

        if self.company_ids:
            conditions.append(
                f"{prefix}company_id = ANY(:f_company_ids)"
            )
            params["f_company_ids"] = list(self.company_ids)

        if self.document_types:
            conditions.append(
                f"{prefix}document_type = ANY(:f_doc_types)"
            )
            params["f_doc_types"] = list(self.document_types)

        if self.published_after is not None:
            conditions.append(
                f"{prefix}published_on >= :f_published_after"
            )
            params["f_published_after"] = self.published_after

        if self.published_before is not None:
            conditions.append(
                f"{prefix}published_on <= :f_published_before"
            )
            params["f_published_before"] = self.published_before

        if not conditions:
            return "", {}

        return " AND ".join(conditions), params