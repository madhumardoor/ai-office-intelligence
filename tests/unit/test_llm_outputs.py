import pytest
from pydantic import ValidationError

from app.schemas.llm_outputs import (
    Claim,
    EvidenceAnalysis,
    ExtractedSignal,
    RagAnswer,
    RouteDecision,
)


def test_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Claim(
            text="Unsupported claim",
            kind="fact",
            evidence_ids=[],
        )


def test_claim_rejects_malformed_evidence_id() -> None:
    with pytest.raises(ValidationError):
        Claim(
            text="Unsupported claim",
            kind="fact",
            evidence_ids=["https://made-up.example"],
        )


def test_inference_can_never_be_flagged_as_confirmed_fact() -> None:
    with pytest.raises(ValidationError):
        EvidenceAnalysis(
            company="X",
            inference_is_confirmed_fact=True,  # type: ignore[arg-type]
        )


def test_unknown_signal_code_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedSignal(
            company_name="X",
            signal_code="vibes",  # type: ignore[arg-type]
            kind="fact",
            evidence_quote="hello world",
            confidence="low",
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            query_type="postgres",
            sql="DROP TABLE x",  # type: ignore[call-arg]
        )


def test_unknown_tool_rejected_so_model_cannot_invent_tools() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            query_type="multi_source",
            tools_required=["raw_sql"],  # type: ignore[list-item]
        )


def test_signal_date_must_be_iso() -> None:
    with pytest.raises(ValidationError):
        ExtractedSignal(
            company_name="X",
            signal_code="recent_funding",
            kind="fact",
            evidence_quote="raised money",
            event_date="Sept 2",
            confidence="high",
        )


def test_insufficient_evidence_answer_is_valid_with_no_claims() -> None:
    answer = RagAnswer(
        sufficient_evidence=False,
        summary="No evidence found.",
    )

    assert answer.claims == []
    assert answer.confidence == "low"