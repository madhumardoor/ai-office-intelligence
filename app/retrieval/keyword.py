"""Keyword retrieval using PostgreSQL full-text search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.filters import RetrievalFilters


@dataclass
class KeywordResult:
    """A single keyword-search result."""

    chunk_id: object
    content: str
    document_type: str | None
    company_id: object | None
    published_on: object | None
    score: float
    rank: int


class KeywordRetriever:
    """Retrieve document chunks using PostgreSQL full-text search."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int = 10,
    ) -> list[KeywordResult]:
        if not query.strip() or top_k <= 0:
            return []

        filters = filters or RetrievalFilters()

        conditions = [
            "dc.content_tsv @@ websearch_to_tsquery('english', :query)"
        ]

        params: dict[str, object] = {
            "query": query,
            "limit": top_k,
        }

        if filters.document_types:
            conditions.append("dc.document_type = ANY(:document_types)")
            params["document_types"] = list(filters.document_types)

        if filters.company_ids:
            conditions.append("dc.company_id = ANY(:company_ids)")
            params["company_ids"] = list(filters.company_ids)

        if filters.published_after is not None:
            conditions.append(
                "(dc.published_on IS NULL OR dc.published_on > :published_after)"
            )
            params["published_after"] = filters.published_after

        if filters.published_before is not None:
            conditions.append(
                "(dc.published_on IS NULL OR dc.published_on <= :published_before)"
            )
            params["published_before"] = filters.published_before

        where_clause = " AND ".join(conditions)

        sql = text(
            f"""
            SELECT
                dc.id,
                dc.content,
                dc.document_type,
                dc.company_id,
                dc.published_on,
                ts_rank_cd(
                    dc.content_tsv,
                    websearch_to_tsquery('english', :query)
                ) AS score
            FROM document_chunks dc
            WHERE {where_clause}
            ORDER BY score DESC, dc.id
            LIMIT :limit
            """
        )

        async with self.session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.mappings().all()

        return [
            KeywordResult(
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