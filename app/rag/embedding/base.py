"""Embedder protocol and error taxonomy."""

from __future__ import annotations

from typing import Protocol


class EmbeddingError(Exception):
    """Base error."""


class EmbeddingUnavailableError(EmbeddingError):
    """Provider unreachable/rate-limited after retries. Callers must NOT drop data: leave rows pending."""


class DimensionMismatchError(EmbeddingError):
    """Provider returned vectors of a different size than the configured/DB dimension."""


class Embedder(Protocol):
    provider: str
    model: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...