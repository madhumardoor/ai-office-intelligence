from __future__ import annotations

from langchain_core.tools import StructuredTool
from app.tools.news_search import news_search
from app.tools.schemas.inputs import NewsSearchInput, WebSearchInput
from app.tools.search_cache import SearchCache
from app.tools.search_provider import SearchProvider
from app.tools.web_search import web_search

from app.retrieval.hybrid import HybridRetriever
from app.tools.calculator import calculate
from app.tools.company_search import company_search
from app.tools.coworking_search import coworking_search
from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.postgis_search import postgis_search
from app.tools.property_search import property_search
from app.tools.schemas.inputs import (
    CalculatorInput,
    CompanySearchInput,
    CoworkingSearchInput,
    PostGISSearchInput,
    PropertySearchInput,
    SignalLookupInput,
    TenantSearchInput,
    VectorSearchInput,
)
from app.tools.signal_lookup import signal_lookup
from app.tools.tenant_search import tenant_search
from app.tools.vector_search import vector_search


def create_company_search_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = CompanySearchInput.model_validate(kwargs)
        return await company_search(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="company_search",
        description=(
            "Search companies using bounded, read-only filters such as "
            "name, domain, industry, city, and employee count."
        ),
        args_schema=CompanySearchInput,
    )


def create_tenant_search_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = TenantSearchInput.model_validate(kwargs)
        return await tenant_search(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="tenant_search",
        description=(
            "Search coworking tenants using company, branch, text query, "
            "and minimum seats used."
        ),
        args_schema=TenantSearchInput,
    )


def create_coworking_search_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = CoworkingSearchInput.model_validate(kwargs)
        return await coworking_search(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="coworking_search",
        description=(
            "Search coworking branches using operator, city, and area "
            "filters."
        ),
        args_schema=CoworkingSearchInput,
    )


def create_property_search_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = PropertySearchInput.model_validate(kwargs)
        return await property_search(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="property_search",
        description=(
            "Search properties using property type, city, area, and "
            "total-area filters."
        ),
        args_schema=PropertySearchInput,
    )


def create_postgis_search_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = PostGISSearchInput.model_validate(kwargs)
        return await postgis_search(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="postgis_search",
        description=(
            "Find nearby company locations using latitude, longitude, "
            "and a bounded radius."
        ),
        args_schema=PostGISSearchInput,
    )


def create_vector_search_tool(
    retriever: HybridRetriever,
) -> StructuredTool:
    async def run(**kwargs):
        request = VectorSearchInput.model_validate(kwargs)
        return await vector_search(request, retriever)

    return StructuredTool.from_function(
        coroutine=run,
        name="vector_search",
        description=(
            "Search indexed documents using bounded hybrid vector and "
            "keyword retrieval."
        ),
        args_schema=VectorSearchInput,
    )


def create_signal_lookup_tool(
    db: ReadOnlyDatabase,
) -> StructuredTool:
    async def run(**kwargs):
        request = SignalLookupInput.model_validate(kwargs)
        return await signal_lookup(request, db)

    return StructuredTool.from_function(
        coroutine=run,
        name="signal_lookup",
        description=(
            "Look up company signals using bounded read-only filters."
        ),
        args_schema=SignalLookupInput,
    )


def create_calculator_tool() -> StructuredTool:
    def run(**kwargs):
        request = CalculatorInput.model_validate(kwargs)
        return calculate(request)

    return StructuredTool.from_function(
        func=run,
        name="calculator",
        description=(
            "Perform safe arithmetic using numbers, parentheses, "
            "addition, subtraction, multiplication, division, modulo, "
            "and unary plus/minus."
        ),
        args_schema=CalculatorInput,
    )

def create_web_search_tool(
    provider: SearchProvider,
    cache: SearchCache,
) -> StructuredTool:
    async def run(**kwargs):
        request = WebSearchInput.model_validate(kwargs)
        return await web_search(request, provider, cache)

    return StructuredTool.from_function(
        coroutine=run,
        name="web_search",
        description=(
            "Search the web using a bounded external-search provider. "
            "Returns titles, URLs, snippets, and source metadata."
        ),
        args_schema=WebSearchInput,
    )


def create_news_search_tool(
    provider: SearchProvider,
    cache: SearchCache,
) -> StructuredTool:
    async def run(**kwargs):
        request = NewsSearchInput.model_validate(kwargs)
        return await news_search(request, provider, cache)

    return StructuredTool.from_function(
        coroutine=run,
        name="news_search",
        description=(
            "Search news using a bounded external-search provider, "
            "optionally filtered by company and city."
        ),
        args_schema=NewsSearchInput,
    )