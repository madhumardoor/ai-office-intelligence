from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load the project's .env explicitly. This matters for CLI scripts because
# os.getenv() does not automatically read a .env file.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


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


class SerperSearchError(RuntimeError):
    """Raised when Serper cannot complete a search request."""


class SerperSearchProvider:
    """
    Real Google SERP provider backed by Serper.

    Required:
        SERPER_API_KEY

    Optional:
        SERPER_GL=IN
        SERPER_HL=en
        SERPER_TIMEOUT_SECONDS=12

    Serper exposes real-time Google search results. We keep this class
    behind the existing SearchProvider protocol so tests can continue
    using MockSearchProvider.
    """

    SEARCH_URL = "https://google.serper.dev/search"
    NEWS_URL = "https://google.serper.dev/news"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        country: str | None = None,
        language: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("SERPER_API_KEY", "")).strip()
        self.country = (country or os.getenv("SERPER_GL", "IN")).strip() or "IN"
        self.language = (language or os.getenv("SERPER_HL", "en")).strip() or "en"

        raw_timeout = timeout_seconds
        if raw_timeout is None:
            raw_timeout = os.getenv("SERPER_TIMEOUT_SECONDS", "12")

        try:
            self.timeout_seconds = max(2.0, min(float(raw_timeout), 30.0))
        except (TypeError, ValueError):
            self.timeout_seconds = 12.0

        if not self.api_key:
            raise SerperSearchError(
                "SERPER_API_KEY is not configured. Add it to your .env file."
            )

        self.web_calls = 0
        self.news_calls = 0

    async def web_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        self.web_calls += 1
        query = query.strip()
        limit = max(1, min(int(limit), 20))

        payload = {
            "q": query,
            "num": limit,
            "gl": self.country,
            "hl": self.language,
        }

        data = await self._request_json(self.SEARCH_URL, payload)

        results: list[SearchResult] = []
        for item in data.get("organic", [])[:limit]:
            if not isinstance(item, dict):
                continue

            url = str(item.get("link", "")).strip()
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()

            if not url or not title:
                continue

            results.append(
                SearchResult(
                    title=title[:500],
                    url=url[:2000],
                    snippet=snippet[:2000],
                    source=self._source_from_item(item),
                    published_at=self._published_from_item(item),
                )
            )

        return results

    async def news_search(
        self,
        query: str,
        company: str | None,
        city: str | None,
        limit: int,
    ) -> list[SearchResult]:
        self.news_calls += 1
        query = query.strip()
        limit = max(1, min(int(limit), 20))

        context_parts = [
            value.strip()
            for value in (company, city)
            if value and value.strip()
        ]
        if context_parts:
            query = f"{query} {' '.join(context_parts)}"

        payload = {
            "q": query,
            "num": limit,
            "gl": self.country,
            "hl": self.language,
        }

        data = await self._request_json(self.NEWS_URL, payload)

        results: list[SearchResult] = []
        for item in data.get("news", [])[:limit]:
            if not isinstance(item, dict):
                continue

            url = str(item.get("link", "")).strip()
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()

            if not url or not title:
                continue

            results.append(
                SearchResult(
                    title=title[:500],
                    url=url[:2000],
                    snippet=snippet[:2000],
                    source=self._source_from_item(item),
                    published_at=self._published_from_item(item),
                )
            )

        return results

    async def _request_json(
        self,
        url: str,
        payload: dict[str, object],
    ) -> dict:
        return await asyncio.to_thread(
            self._request_json_sync,
            url,
            payload,
        )

    def _request_json_sync(
        self,
        url: str,
        payload: dict[str, object],
    ) -> dict:
        body = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AI-Office-Intelligence/1.0",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8", errors="ignore")[:500]
            except Exception:
                pass
            raise SerperSearchError(
                f"Serper HTTP {exc.code}: {details or exc.reason}"
            ) from exc
        except URLError as exc:
            raise SerperSearchError(
                f"Serper network error: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise SerperSearchError("Serper request timed out") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SerperSearchError("Serper returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise SerperSearchError("Serper returned an unexpected response format")

        return data

    @staticmethod
    def _source_from_item(item: dict) -> str | None:
        for key in ("source", "sourceName", "domain"):
            value = item.get(key)
            if value:
                return str(value)[:300]
        return None

    @staticmethod
    def _published_from_item(item: dict) -> str | None:
        for key in ("date", "publishedAt", "published_at"):
            value = item.get(key)
            if value:
                return str(value)[:120]
        return None


class MockSearchProvider:
    """
    Deterministic provider for unit/integration tests.

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
                snippet=f"Mock news search result for query: {query}{suffix}",
                source="mock",
                published_at="2026-09-26",
            )
            for index in range(1, min(limit, 20) + 1)
        ]
