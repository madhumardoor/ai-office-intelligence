"""Plain data types shared across the RAG pipeline (no I/O, no ORM)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(slots=True)
class ParsedDocument:
    """Text extracted from one source, before cleaning/chunking."""

    title: str
    text: str
    source_url: str | None = None
    document_type: str = "other"
    published_on: date | None = None
    location: str | None = None
    company_name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    index: int
    content: str
    content_hash: str
    token_count: int


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with provenance for citations."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    company_id: uuid.UUID | None
    content: str
    title: str
    source_url: str | None
    document_type: str | None
    published_on: date | None
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    citation_id: str | None = None