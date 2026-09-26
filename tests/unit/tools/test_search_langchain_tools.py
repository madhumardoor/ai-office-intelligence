from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools.langchain import (
    create_news_search_tool,
    create_web_search_tool,
)
from app.tools.search_cache import SearchCache
from app.tools.search_provider import MockSearchProvider


@pytest.mark.asyncio
async def test_web_search_langchain_tool():
    provider = MockSearchProvider()
    cache = SearchCache()

    tool = create_web_search_tool(provider, cache)

    result = await tool.ainvoke(
        {
            "query": "Whitefield expansion",
            "limit": 3,
        }
    )

    assert len(result) == 3
    assert provider.web_calls == 1
    assert result[0]["source"] == "mock"


@pytest.mark.asyncio
async def test_news_search_langchain_tool():
    provider = MockSearchProvider()
    cache = SearchCache()

    tool = create_news_search_tool(provider, cache)

    result = await tool.ainvoke(
        {
            "query": "expansion",
            "company": "Nova Labs",
            "city": "Bengaluru",
            "limit": 2,
        }
    )

    assert len(result) == 2
    assert provider.news_calls == 1
    assert "Nova Labs" in result[0]["title"]
    assert "Bengaluru" in result[0]["title"]


@pytest.mark.asyncio
async def test_web_search_rejects_extra_arguments():
    provider = MockSearchProvider()
    cache = SearchCache()

    tool = create_web_search_tool(provider, cache)

    with pytest.raises((ValidationError, ValueError)):
        await tool.ainvoke(
            {
                "query": "Bangalore",
                "limit": 5,
                "execute_sql": "DROP TABLE companies",
            }
        )


def test_search_tools_have_strict_schemas():
    provider = MockSearchProvider()
    cache = SearchCache()

    web_tool = create_web_search_tool(provider, cache)
    news_tool = create_news_search_tool(provider, cache)

    assert web_tool.name == "web_search"
    assert news_tool.name == "news_search"
    assert web_tool.args_schema is not None
    assert news_tool.args_schema is not None