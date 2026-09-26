"""CLI: run a hybrid search against indexed documents.

Usage:
    uv run python -m scripts.search_docs "Which companies are hiring in Whitefield?"
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.rag.embedding.factory import create_embedder
from app.retrieval.context import build_context
from app.retrieval.hybrid import HybridRetriever


async def main(query: str) -> None:
    settings = get_settings()
    engine = create_engine(settings)

    try:
        # Current HybridRetriever API.
        retriever = HybridRetriever(
            create_session_factory(engine),
            create_embedder(settings),
        )

        result = await retriever.retrieve(
            query,
            top_k=5,
        )

        print("=" * 80)
        print("HYBRID RETRIEVAL RESULT")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Methods used: {result.methods_used}")
        print(f"Degraded: {result.degraded}")
        print(f"Results found: {len(result.chunks)}")

        if not result.chunks:
            print("\nNo matching documents found.")
            return

        print("\nRetrieved chunks:")
        print("-" * 80)

        for index, chunk in enumerate(result.chunks, start=1):
            print(f"\n{index}.")
            print(f"   chunk_id: {chunk.chunk_id}")
            print(f"   document_type: {chunk.document_type}")
            print(f"   company_id: {chunk.company_id}")
            print(f"   published_on: {chunk.published_on}")
            print(f"   score: {chunk.score}")
            print(f"   vector_rank: {chunk.vector_rank}")
            print(f"   keyword_rank: {chunk.keyword_rank}")
            print(f"   content: {chunk.content[:500]!r}")

        print("\n" + "=" * 80)
        print("CITATION / CONTEXT")
        print("=" * 80)

        context = build_context(result.chunks)
        print(context.text)

        # Print citation IDs / metadata when available.
        if hasattr(context, "citation_ids"):
            print("\nCitation IDs:")
            print(context.citation_ids)

        if hasattr(context, "stale_ids"):
            print("\nStale evidence IDs:")
            print(context.stale_ids)

    finally:
        await engine.dispose()


def cli() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python -m scripts.search_docs "your question"'
        )

    query = " ".join(sys.argv[1:]).strip()

    if not query:
        raise SystemExit("Query cannot be empty.")

    asyncio.run(main(query))


if __name__ == "__main__":
    cli()