from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_LIMIT = 50
MAX_QUERY_LENGTH = 200


class StrictInput(BaseModel):
    """Base class for every Phase 6 tool input."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class CompanySearchInput(StrictInput):
    """Search the PII-free v_companies view."""

    query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    city: str | None = Field(default=None, max_length=100)
    industry: str | None = Field(default=None, max_length=100)
    min_employee_count: int | None = Field(default=None, ge=0)
    max_employee_count: int | None = Field(default=None, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)

    @field_validator("max_employee_count")
    @classmethod
    def validate_employee_range(cls, value: int | None, info) -> int | None:
        minimum = info.data.get("min_employee_count")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("max_employee_count must be >= min_employee_count")
        return value


class TenantSearchInput(StrictInput):
    """Search coworking tenants."""

    query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    company_id: uuid.UUID | None = None
    branch_id: uuid.UUID | None = None
    min_seats_used: int | None = Field(default=None, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)


class CoworkingSearchInput(StrictInput):
    """Search coworking branches."""

    operator_name: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    area: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)


class PropertySearchInput(StrictInput):
    """Search the PII-free property view."""

    query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    property_type: str | None = Field(default=None, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    area: str | None = Field(default=None, max_length=100)
    min_area_sqft: int | None = Field(default=None, ge=0)
    max_area_sqft: int | None = Field(default=None, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)

    @field_validator("max_area_sqft")
    @classmethod
    def validate_area_range(cls, value: int | None, info) -> int | None:
        minimum = info.data.get("min_area_sqft")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("max_area_sqft must be >= min_area_sqft")
        return value


class PostGISSearchInput(StrictInput):
    """Find companies/locations within a geographic radius."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=100)
    city: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)


class VectorSearchInput(StrictInput):
    """Search indexed document chunks."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    company_id: uuid.UUID | None = None
    document_type: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=8, ge=1, le=20)


class SignalLookupInput(StrictInput):
    """Read structured company signals."""

    company_id: uuid.UUID | None = None
    signal_code: str | None = Field(default=None, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    kind: Literal["fact", "signal", "inference"] | None = None
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)

    @field_validator("signal_code")
    @classmethod
    def validate_signal_code(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not re.fullmatch(r"[a-z0-9_:-]+", value.casefold()):
            raise ValueError(
                "signal_code may contain only letters, numbers, underscores, "
                "hyphens, and colons"
            )

        return value


class CalculatorInput(StrictInput):
    """Restricted arithmetic expression."""

    expression: str = Field(min_length=1, max_length=200)

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, value: str) -> str:
        # This is only an input-level guard.
        # Actual safe AST validation happens inside the calculator tool.
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", value):
            raise ValueError(
                "expression contains unsupported characters"
            )

        return value


class WebSearchInput(StrictInput):
    """Generic web-search request."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    limit: int = Field(default=10, ge=1, le=20)


class NewsSearchInput(StrictInput):
    """News-search request."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    company: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=10, ge=1, le=20)