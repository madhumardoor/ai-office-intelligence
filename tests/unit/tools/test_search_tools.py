from __future__ import annotations

import pytest

from app.tools.news_search import news_search
from app.tools.schemas.inputs import NewsSearchInput, WebSearchInput
from app.tools.search_cache import SearchCache
from app.tools.search_provider import MockSearchProvider
from app.tools.web_search import web_search


@pytest.mark.asyncio
async def test_web_search_uses_cache():
    provider = MockSearchProvider()
    cache = SearchCache(max_entries=10, ttl_seconds=300)

    request = WebSearchInput(
        query="  Whitefield offices  ",
        limit=3,
    )

    first = await web_search(request, provider, cache)
    second = await web_search(request, provider, cache)

    assert first == second
    assert len(first) == 3
    assert provider.web_calls == 1


@pytest.mark.asyncio
async def test_news_search_uses_company_and_city_in_cache_key():
    provider = MockSearchProvider()
    cache = SearchCache(max_entries=10, ttl_seconds=300)

    first_request = NewsSearchInput(
        query="expansion",
        company="Nova Labs",
        city="Bengaluru",
        limit=2,
    )

    second_request = NewsSearchInput(
        query="expansion",
        company="Nova Labs",
        city="Hyderabad",
        limit=2,
    )

    first = await news_search(
        first_request,
        provider,
        cache,
    )

    second = await news_search(
        second_request,
        provider,
        cache,
    )

    assert first != second
    assert provider.news_calls == 2


@pytest.mark.asyncio
async def test_search_results_are_bounded_by_request_limit():
    provider = MockSearchProvider()
    cache = SearchCache()

    request = WebSearchInput(
        query="Bangalore",
        limit=4,
    )

    results = await web_search(
        request,
        provider,
        cache,
    )

    assert len(results) == 4


def test_search_cache_is_bounded():
    cache = SearchCache(
        max_entries=2,
        ttl_seconds=300,
    )

    cache.set(("a",), [{"value": 1}])
    cache.set(("b",), [{"value": 2}])
    cache.set(("c",), [{"value": 3}])

    assert cache.size == 2
    assert cache.get(("a",)) is None
    assert cache.get(("b",)) == [{"value": 2}]
    assert cache.get(("c",)) == [{"value": 3}]