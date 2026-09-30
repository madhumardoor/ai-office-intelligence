from __future__ import annotations

import asyncio
from uuid import UUID

from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.rag.embedding.factory import create_embedder
from app.retrieval.filters import RetrievalFilters
from app.retrieval.hybrid import HybridRetriever


COMPANY_ID = UUID("32477646-ee0b-4bae-938c-3423e7129538")
QUERY = "what does Login Realty do"


async def main() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        retriever = HybridRetriever(
            session_factory,
            create_embedder(settings),
        )

        result = await retriever.retrieve(
            QUERY,
            filters=RetrievalFilters(company_ids=(COMPANY_ID,)),
            top_k=5,
        )

        print("=" * 72)
        print("COMPANY RETRIEVAL TEST")
        print("=" * 72)
        print(f"Company ID : {COMPANY_ID}")
        print(f"Query      : {QUERY}")
        print(f"Chunks     : {len(result.chunks)}")
        print(f"Methods    : {result.methods_used}")
        print(f"Degraded   : {result.degraded}")
        print()

        for index, chunk in enumerate(result.chunks, start=1):
            print(f"[{index}] score={chunk.score:.6f}")
            print(f"company_id={chunk.company_id}")
            print(chunk.content[:500].replace("\n", " "))
            print("-" * 72)

        if result.chunks:
            print("PASS: company-scoped retrieval returned results.")
        else:
            print("FAIL: no company-scoped results returned.")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
