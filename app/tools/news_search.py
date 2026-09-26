from __future__ import annotations

from typing import Any

from app.tools.schemas.inputs import NewsSearchInput
from app.tools.search_cache import SearchCache
from app.tools.search_provider import SearchProvider


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = " ".join(value.strip().casefold().split())

    return normalized or None


async def news_search(
    request: NewsSearchInput,
    provider: SearchProvider,
    cache: SearchCache,
) -> list[dict[str, Any]]:
    key = (
        "news",
        _normalize(request.query),
        _normalize(request.company),
        _normalize(request.city),
        request.limit,
    )

    cached = cache.get(key)

    if cached is not None:
        return cached

    results = await provider.news_search(
        query=request.query.strip(),
        company=request.company.strip() if request.company else None,
        city=request.city.strip() if request.city else None,
        limit=request.limit,
    )

    output = [
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "source": result.source,
            "published_at": result.published_at,
        }
        for result in results[: request.limit]
    ]

    cache.set(key, output)

    return output