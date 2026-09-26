from __future__ import annotations

from typing import Any

from app.retrieval.filters import RetrievalFilters
from app.retrieval.hybrid import HybridRetriever
from app.tools.schemas import VectorSearchInput


async def vector_search(
    request: VectorSearchInput,
    retriever: HybridRetriever,
) -> list[dict[str, Any]]:
    """Run bounded hybrid retrieval through the existing Phase 4 retriever."""

    filters = RetrievalFilters(
        company_ids=(request.company_id,)
        if request.company_id is not None
        else (),
        document_types=(request.document_type,)
        if request.document_type is not None
        else (),
    )

    result = await retriever.retrieve(
        request.query,
        filters=filters,
        top_k=request.limit,
    )

    return [
        {
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "company_id": (
                str(chunk.company_id)
                if chunk.company_id is not None
                else None
            ),
            "title": chunk.title,
            "content": chunk.content,
            "source_url": chunk.source_url,
            "document_type": chunk.document_type,
            "published_on": (
                chunk.published_on.isoformat()
                if chunk.published_on is not None
                else None
            ),
            "score": chunk.score,
            "vector_rank": chunk.vector_rank,
            "keyword_rank": chunk.keyword_rank,
        }
        for chunk in result.chunks
    ]