"""RAG store: documents, chunks (vector + full-text), and embedding metadata."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.enums import DocumentType, check_in
from app.models.mixins import ProvenanceMixin, TimestampMixin, UUIDPrimaryKeyMixin

EMBEDDING_DIM = 384  # local bge-small / MiniLM class. Changing requires a migration.


class Document(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """A source document. `content_hash` dedupes identical content across re-ingestion."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(check_in("document_type", DocumentType), name="document_type_valid"),
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        Index("ix_documents_company_type", "company_id", "document_type"),
    )

    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(30), nullable=False, default=DocumentType.OTHER)
    source_url: Mapped[str | None] = mapped_column(Text)
    published_on: Mapped[date | None] = mapped_column(Date)
    location: Mapped[str | None] = mapped_column(String(150))
    company_name: Mapped[str | None] = mapped_column(String(300))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A retrievable passage.

    - `embedding`: dense vector for semantic search (HNSW cosine index).
    - `content_tsv`: generated tsvector for keyword search (GIN index) -> hybrid retrieval.
    Denormalised company_id / document_type / published_on enable cheap metadata pre-filtering.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_idx"),
        Index("ix_document_chunks_company", "company_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_tsv", "content_tsv", postgresql_using="gin"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="SET NULL")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_count: Mapped[int | None] = mapped_column(Integer)
    document_type: Mapped[str | None] = mapped_column(String(30))
    published_on: Mapped[date | None] = mapped_column(Date)
    embedding = mapped_column(Vector(EMBEDDING_DIM))  # NULL until the embedding job runs
    embedding_model: Mapped[str | None] = mapped_column(String(120))
    content_tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))


class EmbeddingMetadata(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Registry of embedding models used. Detects dimension/model mismatch instead of failing silently."""

    __tablename__ = "embedding_metadata"
    __table_args__ = (UniqueConstraint("provider", "model_name", name="uq_embedding_metadata_model"),)

    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True, server_default="true")