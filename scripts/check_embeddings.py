"""Loads the REAL embedding model once and checks it works. Downloads ~130 MB on first run.

uv run python -m scripts.check_embeddings
"""

from __future__ import annotations

import asyncio
import sys
import time

from app.config import get_settings
from app.rag.embedding.factory import create_embedder


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def main() -> int:
    s = get_settings()
    print(f"provider={s.embedding_provider} model={s.embedding_model} expected_dim={s.embedding_dimension}")
    emb = create_embedder(s)
    t0 = time.perf_counter()
    try:
        docs = await emb.embed_documents(
            [
                "The company is hiring 25 engineers in Whitefield, Bangalore.",
                "Quarterly tax filing deadline for small businesses.",
            ]
        )
        query = await emb.embed_query("Which companies are hiring in Bangalore?")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(f"loaded and encoded in {time.perf_counter() - t0:.1f}s")
    dims = {len(v) for v in [*docs, query]}
    print(f"vector sizes: {dims}")
    if dims != {s.embedding_dimension}:
        print("FAIL: dimension does not match EMBEDDING_DIMENSION")
        return 1
    relevant, unrelated = _dot(query, docs[0]), _dot(query, docs[1])
    print(f"similarity to hiring doc: {relevant:.3f}   to tax doc: {unrelated:.3f}")
    if relevant <= unrelated:
        print("FAIL: relevant document did not score higher than the unrelated one")
        return 1
    print("OK: model works and ranks the relevant document higher")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))