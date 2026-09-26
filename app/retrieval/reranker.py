"""Lightweight deterministic reranking for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankedResult:
    """A retrieval result with its final reranking score."""

    result: object
    score: float


class Reranker:
    """Rerank candidates using lexical overlap with the query.

    This is intentionally deterministic and dependency-light. A more
    sophisticated cross-encoder can replace this component later without
    changing the retrieval interface.
    """

    def rerank(self, query: str, results: list[object]) -> list[object]:
        if not results:
            return []

        query_terms = {
            word.lower()
            for word in query.split()
            if word.strip()
        }

        scored: list[RerankedResult] = []

        for result in results:
            content = getattr(result, "content", "") or ""
            content_terms = set(content.lower().split())

            overlap = len(query_terms & content_terms)
            original_score = float(getattr(result, "score", 0.0) or 0.0)

            final_score = original_score + (overlap * 0.01)

            scored.append(
                RerankedResult(
                    result=result,
                    score=final_score,
                )
            )

        scored.sort(
            key=lambda item: (
                item.score,
                -int(getattr(item.result, "rank", 0) or 0),
            ),
            reverse=True,
        )

        return [item.result for item in scored]