"""Document ingestion and indexing for the RAG pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Document, DocumentChunk
from app.models.enums import DocumentType


SUPPORTED = {".txt", ".html", ".htm", ".pdf"}

# Keep chunks reasonably sized while preserving paragraph boundaries.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class IndexResult:
    """Result returned by one indexing operation."""

    documents_created: int = 0
    documents_skipped_duplicate: int = 0
    chunks_created: int = 0
    failures: list[str] = field(default_factory=list)


class DocumentIndexer:
    """Load, clean, deduplicate and chunk source documents."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

        self.chunk_size = int(
            getattr(settings, "rag_chunk_size", DEFAULT_CHUNK_SIZE)
        )
        self.chunk_overlap = int(
            getattr(settings, "rag_chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        )

        if self.chunk_size <= 0:
            self.chunk_size = DEFAULT_CHUNK_SIZE

        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = DEFAULT_CHUNK_OVERLAP

    async def index_path(
        self,
        session: AsyncSession,
        path: str | Path,
        document_type: str = "other",
        company_id: UUID | None = None,
    ) -> IndexResult:
        """Index one document.

        Invalid/unsupported documents are reported through ``failures`` instead
        of being raised to the caller.
        """

        result = IndexResult()
        file_path = Path(path)

        try:
            self._validate_document_type(document_type)

            if not file_path.exists():
                result.failures.append(f"{file_path}: file does not exist")
                return result

            if not file_path.is_file():
                result.failures.append(f"{file_path}: not a file")
                return result

            suffix = file_path.suffix.lower()

            if suffix not in SUPPORTED:
                result.failures.append(
                    f"{file_path}: unsupported file type "
                    f"'{suffix or 'none'}'"
                )
                return result

            text, metadata = self._load_document(file_path)

            text = self._clean_text(text)

            if not text.strip():
                result.failures.append(
                    f"{file_path}: document contains no text"
                )
                return result

            content_hash = self._hash_text(text)

            # Idempotency: identical source content is indexed only once.
            existing = await session.execute(
                select(Document.id)
                .where(Document.content_hash == content_hash)
                .limit(1)
            )

            if existing.scalar_one_or_none() is not None:
                result.documents_skipped_duplicate = 1
                return result

            published_on = metadata.get("published_on")

            document = Document(
                title=metadata.get("title") or file_path.stem,
                document_type=document_type,
                source_url=metadata.get("source_url"),
                published_on=published_on,
                location=metadata.get("location"),
                company_name=metadata.get("company_name"),
                content_hash=content_hash,
                company_id=company_id,
                meta={
                    "filename": file_path.name,
                    "path": str(file_path),
                    "format": suffix.lstrip("."),
                },
            )

            session.add(document)
            await session.flush()

            chunks = self._chunk_text(text)

            for index, chunk_text in enumerate(chunks):
                chunk_hash = self._hash_text(chunk_text)

                chunk = DocumentChunk(
                    document_id=document.id,
                    company_id=company_id,
                    chunk_index=index,
                    content=chunk_text,
                    content_hash=chunk_hash,
                    token_count=self._estimate_tokens(chunk_text),
                    document_type=document_type,
                    published_on=published_on,
                )

                session.add(chunk)
                result.chunks_created += 1

            # Flush DocumentChunk rows so they are visible to subsequent
            # queries using the same transaction/session.
            await session.flush()

            result.documents_created = 1
            return result

        except Exception as exc:
            result.failures.append(
                f"{file_path}: {type(exc).__name__}: {exc}"
            )
            return result

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_document(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self._load_text(path)

        if suffix in {".html", ".htm"}:
            return self._load_html(path)

        if suffix == ".pdf":
            return self._load_pdf(path)

        raise ValueError(f"Unsupported document type: {suffix}")

    def _load_text(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        raw = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        metadata: dict[str, Any] = {
            "title": path.stem,
        }

        metadata.update(self._extract_text_metadata(raw))

        return raw, metadata

    def _load_html(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        raw = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        soup = BeautifulSoup(raw, "html.parser")

        # Remove content that should never enter the RAG corpus.
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "svg",
            ]
        ):
            tag.decompose()

        title = None

        if soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(" ", strip=True)

        published_on = None

        # Standard article metadata.
        meta_date = soup.find(
            "meta",
            attrs={"property": "article:published_time"},
        )

        if meta_date and meta_date.get("content"):
            published_on = self._parse_date(meta_date["content"])

        # Other common publication metadata.
        if published_on is None:
            for attr in (
                "datePublished",
                "date",
                "pubdate",
            ):
                node = soup.find(
                    "meta",
                    attrs={"name": attr},
                )

                if node and node.get("content"):
                    published_on = self._parse_date(
                        node["content"]
                    )

                    if published_on:
                        break

        text = soup.get_text(
            "\n",
            strip=True,
        )

        metadata: dict[str, Any] = {
            "title": title or path.stem,
            "published_on": published_on,
        }

        metadata.update(
            self._extract_text_metadata(text)
        )

        if published_on is not None:
            metadata["published_on"] = published_on

        return text, metadata

    def _load_pdf(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        reader = PdfReader(str(path))

        pages: list[str] = []

        for page in reader.pages:
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(page_text)

        text = "\n\n".join(pages)

        metadata: dict[str, Any] = {
            "title": path.stem,
        }

        pdf_metadata = reader.metadata

        if pdf_metadata:
            pdf_title = getattr(
                pdf_metadata,
                "title",
                None,
            )

            if pdf_title:
                metadata["title"] = str(
                    pdf_title
                ).strip()

            creation_date = getattr(
                pdf_metadata,
                "creation_date",
                None,
            )

            if isinstance(creation_date, datetime):
                metadata["published_on"] = (
                    creation_date.date()
                )

        metadata.update(
            self._extract_text_metadata(text)
        )

        return text, metadata

    # ------------------------------------------------------------------
    # Cleaning / metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace while preserving meaningful content."""

        text = text.replace("\x00", " ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = []

        for line in text.split("\n"):
            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(line)

        return "\n".join(lines).strip()

    @staticmethod
    def _extract_text_metadata(
        text: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        # Examples:
        # Published: 2026-09-02
        # Published: 2025-01-15
        # Published: 2026-08-20T09:00:00Z
        published_match = re.search(
            r"\bPublished\s*:\s*"
            r"([0-9]{4}-[0-9]{2}-[0-9]{2}"
            r"(?:[T ][0-9]{2}:[0-9]{2}"
            r"(?::[0-9]{2})?"
            r"(?:Z|[+-][0-9]{2}:?[0-9]{2})?)?)",
            text,
            flags=re.IGNORECASE,
        )

        if published_match:
            parsed = DocumentIndexer._parse_date(
                published_match.group(1)
            )

            if parsed:
                metadata["published_on"] = parsed

        # Lightweight location extraction for the synthetic/news corpus.
        locations = [
            "Whitefield",
            "Koramangala",
            "Bellandur",
            "Indiranagar",
            "HSR Layout",
            "Electronic City",
            "Marathahalli",
            "Hebbal",
            "Bengaluru",
            "Bangalore",
        ]

        for location in locations:
            if re.search(
                rf"\b{re.escape(location)}\b",
                text,
                re.IGNORECASE,
            ):
                metadata["location"] = location
                break

        return metadata

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> date | None:
        if not value:
            return None

        value = value.strip()

        # ISO date.
        try:
            return date.fromisoformat(
                value[:10]
            )
        except ValueError:
            pass

        # ISO timestamp with Z.
        try:
            normalized = value.replace(
                "Z",
                "+00:00",
            )

            return datetime.fromisoformat(
                normalized
            ).date()

        except ValueError:
            pass

        # Common fallback formats.
        for fmt in (
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%b %d, %Y",
        ):
            try:
                return datetime.strptime(
                    value,
                    fmt,
                ).date()

            except ValueError:
                continue

        return None

    @staticmethod
    def _validate_document_type(
        document_type: str,
    ) -> None:
        allowed = {
            item.value
            for item in DocumentType
        }

        if document_type not in allowed:
            raise ValueError(
                f"invalid document_type "
                f"'{document_type}'. "
                f"Allowed values: "
                f"{', '.join(sorted(allowed))}"
            )

    # ------------------------------------------------------------------
    # Hashing / chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(
        text: str,
    ) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _estimate_tokens(
        text: str,
    ) -> int:
        # Conservative approximation used only for metadata.
        return max(
            1,
            len(
                re.findall(
                    r"\S+",
                    text,
                )
            ),
        )

    def _chunk_text(
        self,
        text: str,
    ) -> list[str]:
        """Create deterministic overlapping chunks."""

        paragraphs = [
            p.strip()
            for p in re.split(
                r"\n\s*\n",
                text,
            )
            if p.strip()
        ]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            # A very long paragraph needs hard splitting.
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(
                        current.strip()
                    )
                    current = ""

                start = 0

                while start < len(paragraph):
                    end = min(
                        start + self.chunk_size,
                        len(paragraph),
                    )

                    piece = paragraph[
                        start:end
                    ].strip()

                    if piece:
                        chunks.append(piece)

                    if end >= len(paragraph):
                        break

                    start = max(
                        0,
                        end - self.chunk_overlap,
                    )

                continue

            candidate = (
                paragraph
                if not current
                else f"{current}\n\n{paragraph}"
            )

            if len(candidate) <= self.chunk_size:
                current = candidate

            else:
                if current:
                    chunks.append(
                        current.strip()
                    )

                overlap = ""

                if (
                    self.chunk_overlap > 0
                    and chunks
                ):
                    previous = chunks[-1]

                    overlap = previous[
                        -self.chunk_overlap :
                    ].strip()

                current = (
                    f"{overlap}\n\n{paragraph}".strip()
                    if overlap
                    else paragraph
                )

                if len(current) > self.chunk_size:
                    current = current[
                        : self.chunk_size
                    ].strip()

        if current:
            chunks.append(
                current.strip()
            )

        return [
            chunk
            for chunk in chunks
            if chunk
        ]