"""Vector similarity retrieval using pgvector."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from app.retrieval.filters import RetrievalFilters


@dataclass(frozen=True)
class Hit:
    """Minimal retrieval hit used by rank-fusion tests and utilities.

    Parameters
    ----------
    chunk_id:
        Unique identifier of the retrieved chunk.

    rank:
        One-based rank within the retrieval result list.
    """

    chunk_id: object
    rank: int


@dataclass
class VectorResult:
    """A single vector-search result."""

    chunk_id: object
    content: str
    document_type: str | None
    company_id: object | None
    published_on: object | None
    score: float
    rank: int


class VectorRetriever:
    """Retrieve document chunks using pgvector cosine similarity."""

    def __init__(self, session_factory, embedder) -> None:
        self.session_factory = session_factory
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int = 10,
    ) -> list[VectorResult]:
        """Retrieve the most similar document chunks.

        Results are ranked by pgvector cosine distance.
        """

        if not query.strip() or top_k <= 0:
            return []

        filters = filters or RetrievalFilters()

        try:
            query_vector = await self.embedder.embed_query(query)
        except Exception:
            raise

        conditions = [
            "dc.embedding IS NOT NULL",
        ]

        params: dict[str, object] = {
            "query_vector": (
                "["
                + ",".join(str(x) for x in query_vector)
                + "]"
            ),
            "limit": top_k,
        }

        if filters.document_types:
            conditions.append(
                "dc.document_type = ANY(:document_types)"
            )
            params["document_types"] = list(
                filters.document_types
            )

        if filters.company_ids:
            conditions.append(
                "dc.company_id = ANY(:company_ids)"
            )
            params["company_ids"] = list(
                filters.company_ids
            )

        if filters.published_after is not None:
            conditions.append(
                "("
                "dc.published_on IS NULL "
                "OR dc.published_on > :published_after"
                ")"
            )
            params["published_after"] = (
                filters.published_after
            )

        if filters.published_before is not None:
            conditions.append(
                "("
                "dc.published_on IS NULL "
                "OR dc.published_on <= :published_before"
                ")"
            )
            params["published_before"] = (
                filters.published_before
            )

        where_clause = " AND ".join(conditions)

        sql = text(
            f"""
            SELECT
                dc.id,
                dc.content,
                dc.document_type,
                dc.company_id,
                dc.published_on,
                1 - (
                    dc.embedding <=> CAST(
                        :query_vector AS vector
                    )
                ) AS score
            FROM document_chunks dc
            WHERE {where_clause}
            ORDER BY
                dc.embedding <=> CAST(
                    :query_vector AS vector
                ),
                dc.id
            LIMIT :limit
            """
        )

        async with self.session_factory() as session:
            result = await session.execute(
                sql,
                params,
            )

            rows = result.mappings().all()

        return [
            VectorResult(
                chunk_id=row["id"],
                content=row["content"],
                document_type=row["document_type"],
                company_id=row["company_id"],
                published_on=row["published_on"],
                score=float(row["score"] or 0.0),
                rank=index,
            )
            for index, row in enumerate(rows, start=1)
        ]