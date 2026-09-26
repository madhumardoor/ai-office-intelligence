from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str | None = None
    published_at: str | None = None


class SearchProvider(Protocol):
    async def web_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        ...

    async def news_search(
        self,
        query: str,
        company: str | None,
        city: str | None,
        limit: int,
    ) -> list[SearchResult]:
        ...


class MockSearchProvider:
    """
    Deterministic provider for Phase 6 testing.

    No external network calls are made.
    """

    def __init__(self) -> None:
        self.web_calls = 0
        self.news_calls = 0

    async def web_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        self.web_calls += 1

        query = query.strip()

        return [
            SearchResult(
                title=f"Mock web result {index} for {query}",
                url=f"https://mock.example/web/{index}",
                snippet=f"Mock web search result for query: {query}",
                source="mock",
            )
            for index in range(1, min(limit, 20) + 1)
        ]

    async def news_search(
        self,
        query: str,
        company: str | None,
        city: str | None,
        limit: int,
    ) -> list[SearchResult]:
        self.news_calls += 1

        query = query.strip()

        context_parts = [
            value.strip()
            for value in (company, city)
            if value and value.strip()
        ]

        context = ", ".join(context_parts)

        suffix = f" ({context})" if context else ""

        return [
            SearchResult(
                title=f"Mock news result {index} for {query}{suffix}",
                url=f"https://mock.example/news/{index}",
                snippet=(
                    f"Mock news search result for query: "
                    f"{query}{suffix}"
                ),
                source="mock",
                published_at="2026-09-26",
            )
            for index in range(1, min(limit, 20) + 1)
        ]