from __future__ import annotations

import os

import pytest

from app.agent.graph import build_agent_graph
from app.agent.state import initial_state
from app.config import get_settings
from app.llm.factory import create_llm_client


pytestmark = pytest.mark.integration


class FakeCompanySearchTool:
    name = "company_search"

    async def ainvoke(self, kwargs):
        return [
            {
                "company_id": "demo-1",
                "name": "Nova Labs",
                "city": "Bengaluru",
                "employee_count": 120,
            }
        ]


@pytest.mark.asyncio
async def test_live_graph_runs_end_to_end():
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip(
            "Set RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests."
        )

    settings = get_settings()
    llm = create_llm_client(settings)

    graph = build_agent_graph(
        llm=llm,
        tool_registry={
            "company_search": FakeCompanySearchTool(),
        },
    )

    result = await graph.ainvoke(
        initial_state(
            "Is Nova Labs listed in Bengaluru?"
        )
    )

    assert result["route"] is not None
    assert result["tool_results"]["company_search"]

    assert result["evidence_ids"]
    assert result["evidence_context"]

    assert result["answer"] is not None
    assert result["final_text"]

    assert result["verified"] is True
    assert result["verification_errors"] == []