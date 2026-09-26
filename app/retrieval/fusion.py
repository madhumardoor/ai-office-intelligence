"""Fuse keyword and vector retrieval results using Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class FusionResult:
    """A result produced by rank fusion."""

    result: object
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None


class ReciprocalRankFusion:
    """Combine ranked result lists using Reciprocal Rank Fusion.

    RRF score:

        score = weight / (k + rank)

    where rank starts at 1.
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        self.k = k

    def fuse(
        self,
        keyword_results: list[object],
        vector_results: list[object],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> list[FusionResult]:
        """Fuse keyword and vector result lists.

        This method is kept for compatibility with the existing
        object-oriented API.
        """

        result_lists: dict[str, list[object]] = {
            "keyword": keyword_results,
            "vector": vector_results,
        }

        return self._fuse_lists(
            result_lists,
            weights=weights,
        )

    def _fuse_lists(
        self,
        result_lists: Mapping[str, list[object]],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> list[FusionResult]:
        """Perform reciprocal rank fusion."""

        if not result_lists:
            return []

        weights = weights or {}

        combined: dict[object, FusionResult] = {}

        for method_name, results in result_lists.items():
            weight = float(weights.get(method_name, 1.0))

            if weight < 0:
                raise ValueError(
                    f"Weight for '{method_name}' must be non-negative"
                )

            for rank, result in enumerate(results, start=1):
                chunk_id = getattr(result, "chunk_id", None)

                if chunk_id is None:
                    continue

                score = weight / (self.k + rank)

                if chunk_id not in combined:
                    combined[chunk_id] = FusionResult(
                        result=result,
                        score=score,
                    )
                else:
                    combined[chunk_id].score += score

                # Preserve rank information for the two standard
                # retrieval methods.
                if method_name == "keyword":
                    combined[chunk_id].keyword_rank = rank

                elif method_name == "vector":
                    combined[chunk_id].vector_rank = rank

        fused = list(combined.values())

        # Deterministic ordering:
        #
        # 1. Higher RRF score first.
        # 2. Lower original rank first.
        # 3. Chunk ID string as a stable final tie-breaker.
        fused.sort(
            key=self._sort_key,
            reverse=False,
        )

        return fused

    @staticmethod
    def _sort_key(item: FusionResult) -> tuple[float, int, str]:
        """Return a deterministic sorting key."""

        score_key = -item.score

        ranks = [
            rank
            for rank in (
                item.keyword_rank,
                item.vector_rank,
                getattr(item.result, "rank", None),
            )
            if rank is not None
        ]

        best_rank = min(ranks) if ranks else 0

        chunk_id = str(
            getattr(item.result, "chunk_id", "")
        )

        return (
            score_key,
            best_rank,
            chunk_id,
        )


def rrf_fuse(
    lists: Mapping[str, list[object]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[object, float, dict[str, int]]]:
    """Fuse multiple ranked retrieval lists using RRF.

    Parameters
    ----------
    lists:
        Mapping of retrieval method name to ranked results.

        Example:

            {
                "vector": [Hit(A, 1), Hit(B, 2)],
                "keyword": [Hit(B, 1), Hit(C, 2)],
            }

    k:
        RRF constant. Defaults to 60.

    weights:
        Optional per-method weights.

        Example:

            {
                "vector": 2.0,
                "keyword": 1.0,
            }

    Returns
    -------
    list[tuple]
        Each tuple contains:

        (
            chunk_id,
            fused_score,
            {
                "vector": rank,
                "keyword": rank,
            }
        )

    The list is sorted by:

        1. highest fused score
        2. best original rank
        3. chunk ID

    This makes tie-breaking deterministic.
    """

    if not lists:
        return []

    weights = weights or {}

    for method_name, weight in weights.items():
        if weight < 0:
            raise ValueError(
                f"Weight for '{method_name}' must be non-negative"
            )

    scores: dict[object, float] = {}
    ranks: dict[object, dict[str, int]] = {}

    # Keep the actual result object so we can use its chunk_id.
    results_by_id: dict[object, object] = {}

    for method_name, results in lists.items():
        weight = float(weights.get(method_name, 1.0))

        for rank, result in enumerate(results, start=1):
            chunk_id = getattr(result, "chunk_id", None)

            if chunk_id is None:
                continue

            results_by_id.setdefault(chunk_id, result)

            contribution = weight / (k + rank)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + contribution
            )

            ranks.setdefault(
                chunk_id,
                {},
            )[method_name] = rank

    fused = [
        (
            chunk_id,
            scores[chunk_id],
            ranks[chunk_id],
        )
        for chunk_id in scores
    ]

    def sort_key(
        item: tuple[object, float, dict[str, int]],
    ) -> tuple[float, int, str]:
        chunk_id, score, rank_map = item

        best_rank = min(rank_map.values()) if rank_map else 0

        return (
            -score,
            best_rank,
            str(chunk_id),
        )

    fused.sort(key=sort_key)

    return fused