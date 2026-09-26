from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent.graph import build_agent_graph
from app.agent.state import initial_state
from app.llm.client import LLMClient
from app.llm.mock import MockLLM


@dataclass
class FakeTool:
    name: str
    response: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def ainvoke(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        return self.response


@pytest.mark.asyncio
async def test_agent_graph_runs_full_reasoning_pipeline():
    company_tool = FakeTool(
        "company_search",
        [
            {
                "company_id": "1",
                "name": "Nova Labs",
                "city": "Bengaluru",
            }
        ],
    )

    def structured_fn(system, user, schema):
        if schema.__name__ == "RouteDecision":
            return {
                "query_type": "postgres",
                "tools_required": ["company_search"],
                "city": "Bengaluru",
            }

        return {
            "sufficient_evidence": True,
            "summary": "Nova Labs is listed in Bengaluru.",
            "claims": [
                {
                    "text": "Nova Labs is listed in Bengaluru.",
                    "kind": "fact",
                    "evidence_ids": ["E1"],
                }
            ],
            "confidence": "high",
        }

    mock = MockLLM(structured_fn=structured_fn)
    llm = LLMClient(mock)

    graph = build_agent_graph(
        llm=llm,
        tool_registry={
            "company_search": company_tool,
        },
    )

    result = await graph.ainvoke(
        initial_state(
            "Is Nova Labs in Bengaluru?"
        )
    )

    assert result["route"].tools_required == [
        "company_search"
    ]

    assert result["tool_results"]["company_search"] == [
        {
            "company_id": "1",
            "name": "Nova Labs",
            "city": "Bengaluru",
        }
    ]

    assert result["evidence_ids"] == ["E1"]

    assert result["answer"] is not None
    assert result["answer"].summary == (
        "Nova Labs is listed in Bengaluru."
    )

    assert result["verified"] is True
    assert result["verification_errors"] == []


@pytest.mark.asyncio
async def test_agent_graph_marks_invalid_llm_claim_as_unverified():
    company_tool = FakeTool(
        "company_search",
        [
            {
                "company_id": "1",
                "name": "Nova Labs",
                "employee_count": 120,
            }
        ],
    )

    def structured_fn(system, user, schema):
        if schema.__name__ == "RouteDecision":
            return {
                "query_type": "postgres",
                "tools_required": ["company_search"],
            }

        return {
            "sufficient_evidence": True,
            "summary": "Nova Labs has 500 employees.",
            "claims": [
                {
                    "text": "Nova Labs has 500 employees.",
                    "kind": "fact",
                    "evidence_ids": ["E1"],
                }
            ],
            "confidence": "high",
        }

    mock = MockLLM(structured_fn=structured_fn)
    llm = LLMClient(mock)

    graph = build_agent_graph(
        llm=llm,
        tool_registry={
            "company_search": company_tool,
        },
    )

    result = await graph.ainvoke(
        initial_state(
            "How many employees does Nova Labs have?"
        )
    )

    assert result["answer"] is not None
    assert result["verified"] is False
    assert result["verification_errors"]


@pytest.mark.asyncio
async def test_agent_graph_handles_no_tool_results():
    def structured_fn(system, user, schema):
        if schema.__name__ == "RouteDecision":
            return {
                "query_type": "vector",
                "tools_required": [],
            }

        return {
            "sufficient_evidence": False,
            "summary": "There is insufficient evidence.",
            "claims": [],
            "confidence": "low",
        }

    mock = MockLLM(structured_fn=structured_fn)
    llm = LLMClient(mock)

    graph = build_agent_graph(
        llm=llm,
        tool_registry={},
    )

    result = await graph.ainvoke(
        initial_state("What is happening?")
    )

    assert result["tool_results"] == {}
    assert result["evidence_ids"] == []
    assert result["evidence_context"] == ""
    assert result["answer"].sufficient_evidence is False
    assert result["verified"] is True