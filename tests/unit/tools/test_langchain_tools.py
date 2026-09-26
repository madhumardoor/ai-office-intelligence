from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.tools.calculator import calculate
from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.langchain import (
    create_calculator_tool,
    create_company_search_tool,
    create_coworking_search_tool,
    create_postgis_search_tool,
    create_property_search_tool,
    create_signal_lookup_tool,
    create_tenant_search_tool,
    create_vector_search_tool,
)


def test_all_phase6_tools_have_strict_schemas():
    db = AsyncMock(spec=ReadOnlyDatabase)
    retriever = AsyncMock()

    tools = [
        create_company_search_tool(db),
        create_tenant_search_tool(db),
        create_coworking_search_tool(db),
        create_property_search_tool(db),
        create_postgis_search_tool(db),
        create_vector_search_tool(retriever),
        create_signal_lookup_tool(db),
        create_calculator_tool(),
    ]

    names = {tool.name for tool in tools}

    assert names == {
        "company_search",
        "tenant_search",
        "coworking_search",
        "property_search",
        "postgis_search",
        "vector_search",
        "signal_lookup",
        "calculator",
    }

    for tool in tools:
        assert tool.args_schema is not None


@pytest.mark.asyncio
async def test_company_search_langchain_tool():
    db = AsyncMock(spec=ReadOnlyDatabase)

    db.fetch_all.return_value = [
        {
            "id": "company-1",
            "name": "Nova Labs",
            "domain": "nova.example",
            "website": "https://nova.example",
            "industry": "Software",
            "employee_count": 120,
            "city": "Bengaluru",
            "state": "Karnataka",
        }
    ]

    tool = create_company_search_tool(db)

    result = await tool.ainvoke(
        {
            "query": "Nova",
            "city": "Bengaluru",
            "limit": 5,
        }
    )

    assert result == [
        {
            "id": "company-1",
            "name": "Nova Labs",
            "domain": "nova.example",
            "website": "https://nova.example",
            "industry": "Software",
            "employee_count": 120,
            "city": "Bengaluru",
            "state": "Karnataka",
        }
    ]

    db.fetch_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_company_search_tool_rejects_extra_arguments():
    db = AsyncMock(spec=ReadOnlyDatabase)

    tool = create_company_search_tool(db)

    with pytest.raises((ValidationError, ValueError)):
        await tool.ainvoke(
            {
                "query": "Nova",
                "city": "Bengaluru",
                "limit": 5,
                "drop_table": "companies",
            }
        )


@pytest.mark.asyncio
async def test_signal_lookup_tool_rejects_extra_arguments():
    db = AsyncMock(spec=ReadOnlyDatabase)

    tool = create_signal_lookup_tool(db)

    with pytest.raises((ValidationError, ValueError)):
        await tool.ainvoke(
            {
                "company_id": "00000000-0000-0000-0000-000000000001",
                "limit": 5,
                "sql": "DROP TABLE companies",
            }
        )


def test_calculator_tool_works():
    tool = create_calculator_tool()

    result = tool.invoke(
        {
            "expression": "10 * (5 + 2)",
        }
    )

    assert result == 70