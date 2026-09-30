
from __future__ import annotations

import asyncio
from uuid import UUID

from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.rag.embedding.factory import create_embedder
from app.retrieval.hybrid import HybridRetriever
from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.registry import create_tool_registry
from app.tools.search_cache import SearchCache
from app.tools.search_provider import MockSearchProvider

COMPANY_ID = UUID("32477646-ee0b-4bae-938c-3423e7129538")
QUERY = "what does my company do"


async def main() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        db = ReadOnlyDatabase(settings)
        retriever = HybridRetriever(session_factory, create_embedder(settings))

        registry = create_tool_registry(
            db=db,
            retriever=retriever,
            provider=MockSearchProvider(),
            cache=SearchCache(),
            active_company_id=COMPANY_ID,
        )

        print("=" * 72)
        print("COMPANY VECTOR TOOL TEST")
        print("=" * 72)
        print(f"Company ID : {COMPANY_ID}")
        print(f"Query      : {QUERY}")

        result = await registry["vector_search"].ainvoke(
            {"query": QUERY, "limit": 5}
        )

        print(f"Results    : {len(result)}")
        for index, item in enumerate(result, start=1):
            print(f"\n[{index}]")
            print(item)

        if result:
            print("\nPASS: vector_search tool returned company-scoped results.")
        else:
            print("\nFAIL: vector_search returned no results.")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
