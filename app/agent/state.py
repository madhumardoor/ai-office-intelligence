from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.llm_outputs import RagAnswer, RouteDecision


class AgentState(TypedDict, total=False):
    """
    Shared state passed between Phase 7 LangGraph nodes.

    The state deliberately contains no raw database connection,
    filesystem handle, shell command, or unrestricted execution object.
    """

    question: str

    route: RouteDecision | None

    tool_results: dict[str, list[dict[str, Any]]]

    evidence_context: str

    evidence_ids: list[str]

    answer: RagAnswer | None

    verification_errors: list[str]

    verified: bool

    final_text: str


def initial_state(question: str) -> AgentState:
    question = question.strip()

    if not question:
        raise ValueError("question must not be empty")

    if len(question) > 1500:
        raise ValueError("question exceeds the 1500 character limit")

    return AgentState(
        question=question,
        route=None,
        tool_results={},
        evidence_context="",
        evidence_ids=[],
        answer=None,
        verification_errors=[],
        verified=False,
        final_text="",
    )