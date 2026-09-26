"""Local sentence-transformers embedder (no API key, no per-call cost)."""

from __future__ import annotations

import asyncio
from typing import Any

from app.rag.embedding.base import DimensionMismatchError, EmbeddingUnavailableError

# BGE models are trained to expect this prefix on QUERIES (not documents) for retrieval.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    provider = "local"

    def __init__(self, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        # Loading is blocking I/O + CPU: run off the event loop.
                        self._model = await asyncio.to_thread(SentenceTransformer, self.model)
                    except Exception as exc:  # noqa: BLE001
                        raise EmbeddingUnavailableError(
                            f"cannot load local model {self.model!r}: {type(exc).__name__}: {exc}"
                        ) from exc
                    dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                        self._model, "get_sentence_embedding_dimension"
                    )
                    actual = dim_fn()
                    if actual != self.dimension:
                        raise DimensionMismatchError(
                            f"model {self.model!r} outputs {actual} dims but EMBEDDING_DIMENSION={self.dimension}"
                        )
        return self._model

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        model = await self._ensure()
        vecs = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, batch_size=32)
        return [[float(x) for x in v] for v in vecs]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts)

    async def embed_query(self, text: str) -> list[float]:
        prefix = _BGE_QUERY_PREFIX if "bge" in self.model.lower() else ""
        return (await self._encode([prefix + text]))[0]