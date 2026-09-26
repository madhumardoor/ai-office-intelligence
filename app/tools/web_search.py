from __future__ import annotations

from typing import Any

from app.tools.schemas.inputs import WebSearchInput
from app.tools.search_cache import SearchCache
from app.tools.search_provider import SearchProvider


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


async def web_search(
    request: WebSearchInput,
    provider: SearchProvider,
    cache: SearchCache,
) -> list[dict[str, Any]]:
    key = (
        "web",
        _normalize(request.query),
        request.limit,
    )

    cached = cache.get(key)

    if cached is not None:
        return cached

    results = await provider.web_search(
        query=request.query.strip(),
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