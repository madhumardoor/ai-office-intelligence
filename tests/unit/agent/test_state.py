from __future__ import annotations

import pytest

from app.agent.state import AgentState, initial_state
from app.schemas.llm_outputs import RouteDecision


def test_initial_state_has_safe_defaults():
    state = initial_state("Which companies are hiring in Whitefield?")

    assert state["question"] == "Which companies are hiring in Whitefield?"
    assert state["route"] is None
    assert state["tool_results"] == {}
    assert state["evidence_context"] == ""
    assert state["evidence_ids"] == []
    assert state["answer"] is None
    assert state["verification_errors"] == []
    assert state["verified"] is False
    assert state["final_text"] == ""


def test_initial_state_rejects_empty_question():
    with pytest.raises(ValueError):
        initial_state("   ")


def test_initial_state_rejects_oversized_question():
    with pytest.raises(ValueError):
        initial_state("x" * 1501)


def test_state_accepts_typed_route_decision():
    state = initial_state("Find companies hiring in Bengaluru.")

    route = RouteDecision(
        query_type="postgres",
        tools_required=["company_search"],
        company_names=[],
        city="Bengaluru",
    )

    state["route"] = route

    assert state["route"] == route
    assert state["route"].query_type == "postgres"


def test_state_is_not_exposing_execution_handles():
    state = initial_state("Find office expansion signals.")

    forbidden_keys = {
        "db",
        "session",
        "engine",
        "shell",
        "subprocess",
        "python_exec",
        "filesystem",
    }

    assert forbidden_keys.isdisjoint(state.keys())