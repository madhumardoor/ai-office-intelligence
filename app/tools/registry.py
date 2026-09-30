from __future__ import annotations

from uuid import UUID

from langchain_core.tools import StructuredTool

from app.retrieval.hybrid import HybridRetriever
from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.langchain import (
    create_calculator_tool,
    create_company_search_tool,
    create_coworking_search_tool,
    create_news_search_tool,
    create_postgis_search_tool,
    create_property_search_tool,
    create_signal_lookup_tool,
    create_tenant_search_tool,
    create_vector_search_tool,
    create_web_search_tool,
)
from app.tools.search_cache import SearchCache
from app.tools.search_provider import SearchProvider


TOOL_NAMES = (
    "company_search",
    "tenant_search",
    "coworking_search",
    "property_search",
    "postgis_search",
    "vector_search",
    "signal_lookup",
    "calculator",
    "web_search",
    "news_search",
)


def create_tool_registry(
    db: ReadOnlyDatabase,
    retriever: HybridRetriever,
    provider: SearchProvider,
    cache: SearchCache,
    active_company_id: UUID | None = None,
) -> dict[str, StructuredTool]:
    """Create the controlled tool registry."""

    tools = [
        create_company_search_tool(db),
        create_tenant_search_tool(db),
        create_coworking_search_tool(db),
        create_property_search_tool(db),
        create_postgis_search_tool(db),
        create_vector_search_tool(
            retriever,
            default_company_id=active_company_id,
        ),
        create_signal_lookup_tool(
            db,
            default_company_id=active_company_id,
        ),
        create_calculator_tool(),
        create_web_search_tool(provider, cache),
        create_news_search_tool(provider, cache),
    ]

    registry: dict[str, StructuredTool] = {}

    for tool in tools:
        if tool.name in registry:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        registry[tool.name] = tool

    if set(registry) != set(TOOL_NAMES):
        raise ValueError(
            "Tool registry does not match the approved Phase 6 tool set."
        )

    return registry


def get_tools(
    db: ReadOnlyDatabase,
    retriever: HybridRetriever,
    provider: SearchProvider,
    cache: SearchCache,
    active_company_id: UUID | None = None,
) -> list[StructuredTool]:
    """Return the approved tools in deterministic order."""

    registry = create_tool_registry(
        db=db,
        retriever=retriever,
        provider=provider,
        cache=cache,
        active_company_id=active_company_id,
    )

    return [registry[name] for name in TOOL_NAMES]
