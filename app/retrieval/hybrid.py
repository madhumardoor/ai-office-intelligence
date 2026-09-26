"""Hybrid keyword + vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.vector import VectorRetriever


@dataclass
class HybridChunk:
    """Normalized chunk returned by hybrid retrieval."""

    chunk_id: Any
    content: str
    document_type: str | None
    company_id: Any | None
    published_on: Any | None
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None


@dataclass
class HybridResult:
    """Complete hybrid retrieval response."""

    chunks: list[HybridChunk] = field(default_factory=list)
    methods_used: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


class HybridRetriever:
    """Combine PostgreSQL keyword and pgvector semantic retrieval."""

    def __init__(self, session_factory, embedder) -> None:
        self.session_factory = session_factory
        self.embedder = embedder

        self.keyword = KeywordRetriever(session_factory)
        self.vector = VectorRetriever(session_factory, embedder)
        self.fusion = ReciprocalRankFusion()

    async def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int = 10,
    ) -> HybridResult:
        if not query.strip() or top_k <= 0:
            return HybridResult()

        filters = filters or RetrievalFilters()

        keyword_results = []
        vector_results = []
        methods_used: list[str] = []
        degraded: list[str] = []

        # Keyword retrieval should remain available even if embeddings fail.
        try:
            keyword_results = await self.keyword.retrieve(
                query,
                filters,
                top_k=top_k,
            )

            if keyword_results:
                methods_used.append("keyword")

        except Exception as exc:
            degraded.append(f"keyword_error:{type(exc).__name__}")

        # Vector retrieval is allowed to fail gracefully.
        try:
            vector_results = await self.vector.retrieve(
                query,
                filters,
                top_k=top_k,
            )

            if vector_results:
                methods_used.append("vector")

        except Exception as exc:
            print(f"VECTOR ERROR: {type(exc).__name__}: {exc}")
            degraded.append("vector_unavailable")

        # Nothing was retrieved.
        if not keyword_results and not vector_results:
            return HybridResult(
                chunks=[],
                methods_used=methods_used,
                degraded=degraded,
            )

        fused = self.fusion.fuse(
            keyword_results,
            vector_results,
        )

        chunks: list[HybridChunk] = []

        for item in fused[:top_k]:
            result = item.result

            chunks.append(
                HybridChunk(
                    chunk_id=getattr(result, "chunk_id", None),
                    content=getattr(result, "content", ""),
                    document_type=getattr(result, "document_type", None),
                    company_id=getattr(result, "company_id", None),
                    published_on=getattr(result, "published_on", None),
                    score=item.score,
                    keyword_rank=item.keyword_rank,
                    vector_rank=item.vector_rank,
                )
            )

        return HybridResult(
            chunks=chunks,
            methods_used=methods_used,
            degraded=degraded,
        )