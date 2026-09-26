"""Deterministic fake embedder for TESTS ONLY.

Maps words to a bag-of-hashed-words vector, so texts sharing words are closer in cosine space.
This exercises pipeline mechanics (SQL, filters, fusion, caching). It says NOTHING about real semantic
retrieval quality; that is measured later with a real model.
"""

from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    provider = "fake"
    model = "fake-bow"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)  # noqa: S324 - not security use
            v[h % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)