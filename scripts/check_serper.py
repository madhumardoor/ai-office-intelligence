from __future__ import annotations

import asyncio

from app.tools.search_provider import SerperSearchProvider


async def main() -> None:
    provider = SerperSearchProvider()

    results = await provider.web_search(
        "Login Realty Bengaluru",
        5,
    )

    print(f"Results: {len(results)}")

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result.title}")
        print(f"URL: {result.url}")
        print(f"Snippet: {result.snippet[:300]}")
        if result.source:
            print(f"Source: {result.source}")
        if result.published_at:
            print(f"Published: {result.published_at}")


if __name__ == "__main__":
    asyncio.run(main())
