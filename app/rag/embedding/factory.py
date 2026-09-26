from __future__ import annotations

from app.config import Settings
from app.rag.embedding.base import Embedder


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "local":
        from app.rag.embedding.local import LocalEmbedder

        return LocalEmbedder(settings.embedding_model, settings.embedding_dimension)
    raise ValueError(
        f"embedding provider {settings.embedding_provider!r} is not implemented yet; use EMBEDDING_PROVIDER=local"
    )