"""Company onboarding: resolve a company, ingest website/documents, and embed its knowledge.

This module intentionally reuses the existing DocumentIndexer and EmbeddingService.
It does not create a second RAG/indexing pipeline.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
from uuid import UUID

from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.models.company import Company
from app.models.knowledge import Document
from app.rag.embedding.service import EmbeddingService, EmbeddingStats
from app.rag.indexer import DocumentIndexer, IndexResult


DEFAULT_MAX_PAGES = 12
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_PAGE_BYTES = 5 * 1024 * 1024
LEGAL_SUFFIXES = (
    "private limited",
    "pvt ltd",
    "pvt. ltd",
    "limited",
    "ltd",
    "llp",
    "incorporated",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
)


@dataclass
class WebsitePage:
    url: str
    html: str
    title: str = ""


@dataclass
class CompanyOnboardingResult:
    company_id: UUID
    company_name: str
    website: str | None = None
    pages_discovered: int = 0
    documents_created: int = 0
    documents_skipped_duplicate: int = 0
    chunks_created: int = 0
    embeddings_embedded: int = 0
    embeddings_reused: int = 0
    embedding_pending: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failures


def normalize_company_name(value: str) -> str:
    """Normalize names for deterministic matching."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", value or "")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if text.endswith(f" {suffix}"):
            text = text[: -(len(suffix) + 1)].strip()
            break
    return text


def normalize_url(value: str) -> str:
    """Return a normalized HTTP(S) URL."""
    url = (value or "").strip()
    if not url:
        raise ValueError("website URL cannot be empty")
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"unsupported website URL: {value!r}")
    return url.rstrip("/") + "/"


def extract_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").casefold().strip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if not hostname:
        raise ValueError(f"could not extract domain from {url!r}")
    return hostname


def _same_domain(candidate: str, domain: str) -> bool:
    host = (urlparse(candidate).hostname or "").casefold().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host == domain


def _is_html_url(url: str) -> bool:
    path = urlparse(url).path.casefold()
    blocked = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".mp4", ".mp3", ".avi", ".mov",
    )
    return not path.endswith(blocked)


def _clean_page_links(html: str, base_url: str, domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for node in soup.find_all("a", href=True):
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        absolute, _fragment = urldefrag(absolute)
        parsed = urlparse(absolute)

        if parsed.scheme not in {"http", "https"}:
            continue
        if not _same_domain(absolute, domain):
            continue
        if not _is_html_url(absolute):
            continue

        normalized = absolute.rstrip("/") + "/"
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return links


class CompanyWebsiteCrawler:
    """Small bounded same-domain website crawler."""

    def __init__(
        self,
        *,
        max_pages: int = DEFAULT_MAX_PAGES,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        user_agent: str = "AI Office Intelligence/1.0",
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        self.max_pages = max_pages
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes
        self.user_agent = user_agent

    def _robots(self, root_url: str) -> RobotFileParser:
        parsed = urlparse(root_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            # A robots failure must not crash onboarding; the crawler remains bounded.
            pass
        return parser

    def _fetch(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type", "")).casefold()
            if "html" not in content_type and "text/" not in content_type:
                raise ValueError(f"non-HTML content at {url}: {content_type or 'unknown'}")

            data = response.read(self.max_page_bytes + 1)
            if len(data) > self.max_page_bytes:
                raise ValueError(f"page exceeds {self.max_page_bytes // (1024 * 1024)} MB")

        return data.decode("utf-8", errors="replace")

    def crawl(self, root_url: str) -> tuple[list[WebsitePage], list[str]]:
        root_url = normalize_url(root_url)
        domain = extract_domain(root_url)
        robots = self._robots(root_url)

        queue = [root_url]
        queued = {root_url}
        visited: set[str] = set()
        pages: list[WebsitePage] = []
        failures: list[str] = []

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                if not robots.can_fetch(self.user_agent, url):
                    failures.append(f"{url}: blocked by robots.txt")
                    continue

                html = self._fetch(url)
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.get_text(" ", strip=True) if soup.title else ""
                pages.append(WebsitePage(url=url, html=html, title=title))

                for link in _clean_page_links(html, url, domain):
                    if link not in queued and len(queue) + len(pages) < self.max_pages * 3:
                        queued.add(link)
                        queue.append(link)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{url}: {type(exc).__name__}: {exc}")

        return pages, failures


class CompanyOnboardingService:
    """Create/find a company and feed all supplied sources into the existing RAG stack."""

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        embedder,
        *,
        max_website_pages: int = DEFAULT_MAX_PAGES,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.indexer = DocumentIndexer(settings)
        self.embedder = embedder
        self.embedding_service = EmbeddingService(
            embedder,
            batch_size=settings.embedding_batch_size,
        )
        self.crawler = CompanyWebsiteCrawler(
            max_pages=max_website_pages,
            user_agent="AI Office Intelligence/1.0",
        )

    async def _get_or_create_company(
        self,
        session: AsyncSession,
        name: str,
        website: str | None,
    ) -> Company:
        normalized_name = normalize_company_name(name)
        normalized_url = normalize_url(website) if website else None
        domain = extract_domain(normalized_url) if normalized_url else None

        company = None

        if domain:
            result = await session.execute(
                select(Company)
                .where(
                    Company.domain == domain,
                    Company.deleted_at.is_(None),
                )
                .limit(1)
            )
            company = result.scalar_one_or_none()

        if company is None and normalized_name:
            result = await session.execute(
                select(Company)
                .where(
                    Company.normalized_name == normalized_name,
                    Company.deleted_at.is_(None),
                )
                .limit(1)
            )
            company = result.scalar_one_or_none()

        if company is None:
            company = Company(
                name=name.strip(),
                normalized_name=normalized_name or name.strip().casefold(),
                website=normalized_url,
                domain=domain,
                country="IN",
            )
            session.add(company)
            await session.flush()
        else:
            if normalized_url and not company.website:
                company.website = normalized_url
            if domain and not company.domain:
                company.domain = domain
            if not company.description:
                company.description = None

        return company

    async def _set_document_source_url(
        self,
        session: AsyncSession,
        *,
        content_hash: str,
        source_url: str,
        company_id: UUID,
        extra_meta: dict,
    ) -> None:
        result = await session.execute(
            select(Document)
            .where(
                Document.content_hash == content_hash,
                Document.company_id == company_id,
            )
            .limit(1)
        )
        document = result.scalar_one_or_none()
        if document is None:
            return

        merged_meta = dict(document.meta or {})
        merged_meta.update(extra_meta)
        document.meta = merged_meta
        document.source_url = source_url

    async def _index_temp_file(
        self,
        session: AsyncSession,
        *,
        path: Path,
        company_id: UUID,
        document_type: str,
        source_url: str | None = None,
    ) -> IndexResult:
        result = await self.indexer.index_path(
            session=session,
            path=path,
            document_type=document_type,
            company_id=company_id,
        )

        if result.documents_created and source_url:
            # DocumentIndexer hashes cleaned text, not raw bytes. Recreate its
            # cleaned text path for deterministic source-url attachment.
            text, _meta = self.indexer._load_document(path)  # noqa: SLF001
            cleaned = self.indexer._clean_text(text)  # noqa: SLF001
            content_hash = self.indexer._hash_text(cleaned)  # noqa: SLF001
            await self._set_document_source_url(
                session,
                content_hash=content_hash,
                source_url=source_url,
                company_id=company_id,
                extra_meta={
                    "source": "company_onboarding",
                    "source_url": source_url,
                },
            )

        return result

    async def onboard(
        self,
        *,
        company_name: str,
        website: str | None = None,
        document_paths: Iterable[Path] = (),
    ) -> CompanyOnboardingResult:
        if not company_name.strip():
            raise ValueError("company_name is required")

        normalized_url = normalize_url(website) if website else None
        docs = [Path(p) for p in document_paths]

        result: CompanyOnboardingResult

        async with self.session_factory() as session:
            async with session.begin():
                company = await self._get_or_create_company(
                    session,
                    name=company_name,
                    website=normalized_url,
                )

                result = CompanyOnboardingResult(
                    company_id=company.id,
                    company_name=company.name,
                    website=company.website,
                )

                # 1) Website → bounded same-domain crawl → existing DocumentIndexer.
                if normalized_url:
                    pages, failures = self.crawler.crawl(normalized_url)
                    result.pages_discovered = len(pages)
                    result.failures.extend(failures)

                    with tempfile.TemporaryDirectory(prefix="aoi_web_") as tmp:
                        root = Path(tmp)
                        for index, page in enumerate(pages, start=1):
                            page_path = root / f"page_{index}.html"
                            page_path.write_text(page.html, encoding="utf-8")
                            index_result = await self._index_temp_file(
                                session,
                                path=page_path,
                                company_id=company.id,
                                document_type="company_profile",
                                source_url=page.url,
                            )
                            result.documents_created += index_result.documents_created
                            result.documents_skipped_duplicate += (
                                index_result.documents_skipped_duplicate
                            )
                            result.chunks_created += index_result.chunks_created
                            result.failures.extend(index_result.failures)

                # 2) User-provided documents → existing DocumentIndexer.
                for path in docs:
                    index_result = await self.indexer.index_path(
                        session=session,
                        path=path,
                        document_type="document",
                        company_id=company.id,
                    )
                    result.documents_created += index_result.documents_created
                    result.documents_skipped_duplicate += (
                        index_result.documents_skipped_duplicate
                    )
                    result.chunks_created += index_result.chunks_created
                    result.failures.extend(index_result.failures)

                await session.flush()

                # 3) Embed only this company's chunks.
                embedding_stats: EmbeddingStats = await self.embedding_service.embed_pending(
                    session,
                    company_id=company.id,
                )
                result.embeddings_embedded = embedding_stats.embedded
                result.embeddings_reused = embedding_stats.reused_from_cache
                result.embedding_pending = embedding_stats.pending_after

        return result
