from __future__ import annotations

from app.agent.finalizer import build_final_response
from app.agent.state import initial_state
from app.schemas.llm_outputs import RagAnswer


def make_answer(
    *,
    sufficient_evidence: bool,
    summary: str,
    claims: list[dict],
    caveats: list[str] | None = None,
    confidence: str = "low",
) -> RagAnswer:
    return RagAnswer.model_validate(
        {
            "sufficient_evidence": sufficient_evidence,
            "summary": summary,
            "claims": claims,
            "caveats": caveats or [],
            "confidence": confidence,
        }
    )


def test_finalizer_formats_verified_answer_with_citations():
    state = initial_state(
        "Is Nova Labs expanding in Whitefield?"
    )

    state["answer"] = make_answer(
        sufficient_evidence=True,
        summary="Nova Labs is expanding in Whitefield.",
        claims=[
            {
                "text": "Nova Labs is hiring 25 engineers.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            },
            {
                "text": (
                    "This may indicate additional office-space demand."
                ),
                "kind": "inference",
                "evidence_ids": ["E1", "E2"],
            },
        ],
        caveats=[
            "The expansion signal is based on recent retrieved evidence."
        ],
        confidence="high",
    )

    state["evidence_ids"] = ["E1", "E2"]
    state["verified"] = True
    state["verification_errors"] = []

    result = build_final_response(state)

    assert "Nova Labs is expanding in Whitefield." in result["final_text"]
    assert "- Nova Labs is hiring 25 engineers. [E1]" in result["final_text"]
    assert (
        "- This may indicate additional office-space demand. [E1] [E2]"
        in result["final_text"]
    )
    assert "Caveats:" in result["final_text"]


def test_finalizer_does_not_present_unverified_answer_as_fact():
    state = initial_state(
        "How many employees does Nova Labs have?"
    )

    state["answer"] = make_answer(
        sufficient_evidence=True,
        summary="Nova Labs has 500 employees.",
        claims=[
            {
                "text": "Nova Labs has 500 employees.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ],
        confidence="high",
    )

    state["evidence_ids"] = ["E1"]
    state["verified"] = False
    state["verification_errors"] = [
        "number '500' in claim is not present in cited evidence E1"
    ]

    result = build_final_response(state)

    assert "I could not fully verify" in result["final_text"]
    assert "number '500'" in result["final_text"]
    assert "Nova Labs has 500 employees." not in result["final_text"]


def test_finalizer_handles_insufficient_evidence():
    state = initial_state("What is happening?")

    state["answer"] = make_answer(
        sufficient_evidence=False,
        summary="There is insufficient evidence.",
        claims=[],
        confidence="low",
    )

    state["verified"] = True
    state["verification_errors"] = []
    state["evidence_ids"] = []

    result = build_final_response(state)

    assert result["final_text"] == (
        "There is insufficient evidence."
    )


def test_finalizer_preserves_state():
    state = initial_state(
        "What companies are expanding?"
    )

    state["answer"] = make_answer(
        sufficient_evidence=True,
        summary="Nova Labs is expanding.",
        claims=[
            {
                "text": "Nova Labs is expanding.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ],
        confidence="medium",
    )

    state["verified"] = True
    state["evidence_ids"] = ["E1"]

    result = build_final_response(state)

    assert result["question"] == "What companies are expanding?"
    assert result["verified"] is True
    assert result["evidence_ids"] == ["E1"]
    assert result["final_text"]