from __future__ import annotations

from app.agent.state import initial_state
from app.agent.verifier import verify_answer
from app.schemas.llm_outputs import RagAnswer


def _answer(
    *,
    claims: list[dict],
    sufficient_evidence: bool = True,
) -> RagAnswer:
    return RagAnswer.model_validate(
        {
            "sufficient_evidence": sufficient_evidence,
            "summary": "Test answer.",
            "claims": claims,
            "confidence": "high",
        }
    )


def _state() -> dict:
    state = initial_state(
        "Is Nova Labs expanding in Whitefield?"
    )

    state["evidence_context"] = (
        '{"evidence_id": "E1", '
        '"tool": "vector_search", '
        '"data": {"text": '
        '"Nova Labs is expanding in Whitefield and hiring 25 engineers."}}'
    )

    state["evidence_ids"] = ["E1"]

    return state


def test_verifier_accepts_valid_evidence_reference():
    state = _state()

    state["answer"] = _answer(
        claims=[
            {
                "text": "Nova Labs is hiring 25 engineers.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is True
    assert result["verification_errors"] == []


def test_verifier_rejects_missing_evidence_id():
    state = _state()

    state["answer"] = _answer(
        claims=[
            {
                "text": "Nova Labs is expanding.",
                "kind": "fact",
                "evidence_ids": ["E99"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is False
    assert any(
        "missing evidence ID: E99" in error
        for error in result["verification_errors"]
    )


def test_verifier_rejects_mismatched_number():
    state = _state()

    state["answer"] = _answer(
        claims=[
            {
                "text": "Nova Labs is hiring 50 engineers.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is False
    assert any(
        "number '50'" in error
        for error in result["verification_errors"]
    )


def test_verifier_rejects_missing_quote():
    state = _state()

    state["answer"] = _answer(
        claims=[
            {
                "text": 'The company describes itself as "the fastest growing team".',
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is False
    assert any(
        "quoted text" in error
        for error in result["verification_errors"]
    )


def test_verifier_accepts_existing_quote():
    state = _state()

    state["answer"] = _answer(
        claims=[
            {
                "text": 'The evidence says "Nova Labs is expanding in Whitefield".',
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is True
    assert result["verification_errors"] == []


def test_verifier_rejects_claims_without_evidence():
    state = initial_state("What is happening?")
    state["evidence_context"] = ""
    state["evidence_ids"] = []

    state["answer"] = _answer(
        claims=[
            {
                "text": "Nova Labs is expanding.",
                "kind": "fact",
                "evidence_ids": ["E1"],
            }
        ]
    )

    result = verify_answer(state)

    assert result["verified"] is False
    assert any(
        "no evidence is available" in error
        for error in result["verification_errors"]
    )