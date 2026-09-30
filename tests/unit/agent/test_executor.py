from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent.executor import execute_tools
from app.agent.state import initial_state
from app.schemas.llm_outputs import RouteDecision


@dataclass
class FakeTool:
    name: str
    response: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def ainvoke(
        self,
        kwargs: dict[str, Any],
    ) -> Any:
        self.calls.append(kwargs)
        return self.response


def route(**overrides) -> RouteDecision:
    values = {
        "query_type": "postgres",
        "tools_required": [],
    }

    values.update(overrides)

    return RouteDecision.model_validate(values)


@pytest.mark.asyncio
async def test_executor_runs_only_selected_tools():
    company_tool = FakeTool(
        "company_search",
        [{"company_id": "1", "name": "Nova Labs"}],
    )

    web_tool = FakeTool(
        "web_search",
        [{"title": "web result"}],
    )

    registry = {
        "company_search": company_tool,
        "web_search": web_tool,
    }

    state = initial_state(
        "Which companies are hiring in Bengaluru?"
    )

    state["route"] = route(
        query_type="postgres",
        tools_required=["company_search"],
        city="Bengaluru",
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"]["company_search"] == [
        {
            "company_id": "1",
            "name": "Nova Labs",
        }
    ]

    # City-only question:
    # query must be None so the database searches the city,
    # rather than searching the entire natural-language question
    # as a company name.
    assert company_tool.calls == [
        {
            "query": None,
            "city": "Bengaluru",
            "limit": 20,
        }
    ]

    # Unselected tools must never execute.
    assert web_tool.calls == []


@pytest.mark.asyncio
async def test_executor_uses_explicit_company_name():
    company_tool = FakeTool(
        "company_search",
        [{"company_id": "1", "name": "Nova Labs"}],
    )

    registry = {
        "company_search": company_tool,
    }

    state = initial_state(
        "Is Nova Labs listed in Bengaluru?"
    )

    state["route"] = route(
        query_type="postgres",
        tools_required=["company_search"],
        city="Bengaluru",
        company_names=["Nova Labs"],
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"]["company_search"] == [
        {
            "company_id": "1",
            "name": "Nova Labs",
        }
    ]

    assert company_tool.calls == [
        {
            "query": "Nova Labs",
            "city": "Bengaluru",
            "limit": 20,
        }
    ]


@pytest.mark.asyncio
async def test_executor_builds_vector_search_arguments():
    vector_tool = FakeTool(
        "vector_search",
        [{"chunk_id": "E1", "text": "Nova is expanding"}],
    )

    registry = {
        "vector_search": vector_tool,
    }

    state = initial_state(
        "What are Nova Labs' expansion plans?"
    )

    state["route"] = route(
        query_type="vector",
        tools_required=["vector_search"],
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"]["vector_search"] == [
        {
            "chunk_id": "E1",
            "text": "Nova is expanding",
        }
    ]

    assert vector_tool.calls == [
        {
            "query": "What are Nova Labs' expansion plans?",
            "limit": 8,
        }
    ]


@pytest.mark.asyncio
async def test_executor_skips_postgis_without_coordinates():
    postgis_tool = FakeTool(
        "postgis_search",
        [{"company": "Should not run"}],
    )

    registry = {
        "postgis_search": postgis_tool,
    }

    state = initial_state(
        "Find companies near Whitefield."
    )

    state["route"] = route(
        query_type="postgis",
        tools_required=["postgis_search"],
        city="Bengaluru",
        radius_km=5,
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"]["postgis_search"] == [
        {
            "status": "skipped",
            "reason": (
                "insufficient information "
                "for safe execution"
            ),
        }
    ]

    assert postgis_tool.calls == []


@pytest.mark.asyncio
async def test_executor_runs_calculator_only_with_extractable_expression():
    calculator = FakeTool(
        "calculator",
        [{"result": 200}],
    )

    registry = {
        "calculator": calculator,
    }

    state = initial_state(
        "What is 125 + 75?"
    )

    state["route"] = route(
        query_type="postgres",
        tools_required=["calculator"],
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"]["calculator"] == [
        {
            "result": 200,
        }
    ]

    assert calculator.calls == [
        {
            "expression": "125 + 75",
        }
    ]


@pytest.mark.asyncio
async def test_executor_does_not_execute_tools_when_clarification_needed():
    company_tool = FakeTool(
        "company_search",
        [{"company": "Should not run"}],
    )

    registry = {
        "company_search": company_tool,
    }

    state = initial_state(
        "Which company are you asking about?"
    )

    state["route"] = route(
        query_type="clarification",
        tools_required=["company_search"],
        needs_clarification=True,
        clarification_question="Which company?",
    )

    result = await execute_tools(
        state,
        registry,
    )

    assert result["tool_results"] == {}
    assert company_tool.calls == []