"""Batch embedding service with caching and safe failure handling."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import DocumentChunk
from app.rag.embedding.base import (
    DimensionMismatchError,
    EmbeddingError,
    Embedder,
)


@dataclass(frozen=True)
class EmbeddingStats:
    embedded: int = 0
    reused_from_cache: int = 0
    pending_after: int = 0
    failed_batches: int = 0


class EmbeddingService:
    """Embed pending document chunks without losing data on provider failures."""

    def __init__(self, embedder: Embedder, batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.embedder = embedder
        self.batch_size = batch_size

    async def embed_pending(
        self,
        session: AsyncSession,
        company_id: UUID | None = None,
    ) -> EmbeddingStats:
        """Embed pending chunks, optionally restricted to one company.

        Identical chunk content reuses an existing embedding from the database.
        Provider failures leave affected chunks pending instead of deleting data.
        """

        query = (
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.created_at, DocumentChunk.id)
        )

        if company_id is not None:
            query = query.where(DocumentChunk.company_id == company_id)

        result = await session.execute(query)
        pending = list(result.scalars().all())

        if not pending:
            return EmbeddingStats()

        embedded = 0
        reused = 0
        failed_batches = 0

        # Cache identical content hashes already embedded in the database.
        for chunk in pending:
            cached = await session.execute(
                select(DocumentChunk.embedding, DocumentChunk.embedding_model)
                .where(
                    DocumentChunk.content_hash == chunk.content_hash,
                    DocumentChunk.embedding.is_not(None),
                )
                .limit(1)
            )
            cached_row = cached.first()

            if cached_row is not None:
                chunk.embedding = cached_row[0]
                chunk.embedding_model = cached_row[1] or self.embedder.model
                reused += 1

        # Only send genuinely uncached chunks to the provider.
        remaining = [chunk for chunk in pending if chunk.embedding is None]

        for start in range(0, len(remaining), self.batch_size):
            batch = remaining[start : start + self.batch_size]
            texts = [chunk.content for chunk in batch]

            try:
                vectors = await self.embedder.embed_documents(texts)

                if len(vectors) != len(batch):
                    raise EmbeddingError(
                        f"provider returned {len(vectors)} vectors "
                        f"for {len(batch)} texts"
                    )

                for chunk, vector in zip(batch, vectors):
                    if len(vector) != self.embedder.dimension:
                        raise DimensionMismatchError(
                            f"expected dimension {self.embedder.dimension}, "
                            f"got {len(vector)}"
                        )

                    chunk.embedding = vector
                    chunk.embedding_model = self.embedder.model

                embedded += len(batch)

            except EmbeddingError:
                failed_batches += 1

            except Exception:
                failed_batches += 1

        await session.flush()

        pending_query = select(DocumentChunk.id).where(
            DocumentChunk.embedding.is_(None)
        )
        if company_id is not None:
            pending_query = pending_query.where(
                DocumentChunk.company_id == company_id
            )

        pending_after = (await session.execute(pending_query)).all()

        return EmbeddingStats(
            embedded=embedded,
            reused_from_cache=reused,
            pending_after=len(pending_after),
            failed_batches=failed_batches,
        )
