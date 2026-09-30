import asyncio

from app.tools.search_provider import SerperSearchProvider


async def main():
    provider = SerperSearchProvider()

    results = await provider.web_search(
        "Login Realty founder K.P. Shetty",
        5,
    )

    print(f"\nFound {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"RESULT {i}")
        print(f"TITLE: {result.title}")
        print(f"URL: {result.url}")
        print(f"SNIPPET: {result.snippet}")
        print()


if __name__ == "__main__":
    asyncio.run(main())