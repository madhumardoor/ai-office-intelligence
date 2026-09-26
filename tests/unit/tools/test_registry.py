from __future__ import annotations

from unittest.mock import AsyncMock

from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.registry import TOOL_NAMES, create_tool_registry, get_tools
from app.tools.search_cache import SearchCache
from app.tools.search_provider import MockSearchProvider


def _dependencies():
    db = AsyncMock(spec=ReadOnlyDatabase)
    retriever = AsyncMock()
    provider = MockSearchProvider()
    cache = SearchCache()

    return db, retriever, provider, cache


def test_create_tool_registry_contains_exact_phase6_tool_set():
    db, retriever, provider, cache = _dependencies()

    registry = create_tool_registry(
        db=db,
        retriever=retriever,
        provider=provider,
        cache=cache,
    )

    assert set(registry) == set(TOOL_NAMES)
    assert len(registry) == 10

    for name in TOOL_NAMES:
        assert name in registry
        assert registry[name].name == name
        assert registry[name].args_schema is not None


def test_get_tools_returns_deterministic_order():
    db, retriever, provider, cache = _dependencies()

    tools = get_tools(
        db=db,
        retriever=retriever,
        provider=provider,
        cache=cache,
    )

    assert [tool.name for tool in tools] == list(TOOL_NAMES)


def test_registry_does_not_expose_unapproved_tools():
    db, retriever, provider, cache = _dependencies()

    registry = create_tool_registry(
        db=db,
        retriever=retriever,
        provider=provider,
        cache=cache,
    )

    forbidden_names = {
        "sql_execute",
        "raw_sql",
        "shell",
        "filesystem",
        "python_exec",
        "web_browser",
        "browser",
        "http_request",
    }

    assert forbidden_names.isdisjoint(registry)