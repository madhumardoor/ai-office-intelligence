"""Structured outputs required from the LLM layer.

These schemas are deliberately strict:
- closed vocabularies for routing/tools/signals
- bounded lists
- forbidden extra fields
- evidence IDs required for claims
- inference can never be marked as a confirmed fact
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


QueryType = Literal[
    "postgres",
    "postgis",
    "vector",
    "web",
    "multi_source",
    "clarification",
    "out_of_scope",
]

ToolName = Literal[
    "company_search",
    "tenant_search",
    "coworking_search",
    "property_search",
    "postgis_search",
    "vector_search",
    "news_search",
    "web_search",
    "signal_lookup",
    "calculator",
]


class _Strict(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class RouteDecision(_Strict):
    """Output of the FAST-tier query classifier."""

    query_type: QueryType

    tools_required: list[ToolName] = Field(
        default_factory=list,
        max_length=6,
    )

    needs_clarification: bool = False

    clarification_question: str | None = Field(
        default=None,
        max_length=300,
    )

    company_names: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    city: str | None = None
    area: str | None = None

    radius_km: float | None = Field(
        default=None,
        gt=0,
        le=100,
    )

    min_open_roles: int | None = Field(
        default=None,
        ge=0,
    )

    needs_fresh_web_data: bool = False

    reasoning: str = Field(
        default="",
        max_length=400,
    )

    @field_validator("tools_required")
    @classmethod
    def unique_tools(cls, value: list[ToolName]) -> list[ToolName]:
        """Remove duplicate tool names while preserving order."""
        return list(dict.fromkeys(value))


class Claim(_Strict):
    """One answer statement tied to supporting evidence."""

    text: str = Field(
        max_length=600,
    )

    kind: Literal[
        "fact",
        "signal",
        "inference",
    ]

    # A claim without evidence is structurally invalid.
    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=6,
    )

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        """Evidence references must use E1, E2, E3 ... format."""
        import re

        if not all(
            re.fullmatch(r"E\d{1,3}", evidence_id)
            for evidence_id in value
        ):
            raise ValueError(
                "evidence ids must look like 'E1'"
            )

        return value


class RagAnswer(_Strict):
    """Output of the grounded-answer step."""

    sufficient_evidence: bool

    summary: str = Field(
        max_length=1200,
    )

    claims: list[Claim] = Field(
        default_factory=list,
        max_length=12,
    )

    caveats: list[str] = Field(
        default_factory=list,
        max_length=6,
    )

    confidence: Literal[
        "low",
        "medium",
        "high",
    ] = "low"


class ExtractedCompany(_Strict):
    """One explicitly mentioned company."""

    name: str = Field(
        max_length=300,
    )

    role: Literal[
        "subject",
        "investor",
        "competitor",
        "partner",
        "customer",
        "other",
    ] = "subject"

    # Must be copied from the source text.
    quote: str = Field(
        max_length=400,
    )


class CompanyExtraction(_Strict):
    """Companies explicitly mentioned in a document."""

    companies: list[ExtractedCompany] = Field(
        default_factory=list,
        max_length=15,
    )


SignalCode = Literal[
    "hiring_acceleration",
    "headcount_growth",
    "recent_funding",
    "expansion_announcement",
    "multi_location_presence",
    "leadership_hiring",
]


class ExtractedSignal(_Strict):
    """One office-demand-related signal found in source evidence."""

    company_name: str = Field(
        max_length=300,
    )

    signal_code: SignalCode

    kind: Literal[
        "fact",
        "signal",
    ]

    # Must be copied verbatim from the source text.
    evidence_quote: str = Field(
        min_length=5,
        max_length=500,
    )

    value: float | None = None

    event_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    confidence: Literal[
        "low",
        "medium",
        "high",
    ]


class SignalExtraction(_Strict):
    """Signals extracted from one document."""

    signals: list[ExtractedSignal] = Field(
        default_factory=list,
        max_length=15,
    )


class EvidenceAnalysis(_Strict):
    """FACT -> SIGNAL -> INFERENCE chain for one company."""

    company: str = Field(
        max_length=300,
    )

    facts: list[Claim] = Field(
        default_factory=list,
        max_length=8,
    )

    signals: list[Claim] = Field(
        default_factory=list,
        max_length=8,
    )

    inference: Claim | None = None

    # Schema-level guard:
    # an inference is never allowed to be represented as confirmed fact.
    inference_is_confirmed_fact: Literal[False] = False

    conflicting_evidence: list[str] = Field(
        default_factory=list,
        max_length=5,
    )

    limitations: list[str] = Field(
        default_factory=list,
        max_length=6,
    )