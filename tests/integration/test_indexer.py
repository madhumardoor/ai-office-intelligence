from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import Settings
from app.rag.embedding.base import EmbeddingUnavailableError
from app.rag.embedding.fake import FakeEmbedder
from app.rag.embedding.service import EmbeddingService
from app.rag.indexer import DocumentIndexer

pytestmark = pytest.mark.integration
DOCS = Path(__file__).resolve().parents[2] / "data" / "samples" / "docs"


@pytest.fixture
def indexer() -> DocumentIndexer:
    return DocumentIndexer(Settings(app_env="test"))


async def test_index_sample_docs_with_metadata(db_session, indexer) -> None:
    r = await indexer.index_path(db_session, DOCS / "nova_labs_expansion.txt", document_type="news")
    assert r.documents_created == 1 and r.chunks_created >= 1 and not r.failures
    row = (
        await db_session.execute(
            text("SELECT published_on, location, document_type FROM documents WHERE title LIKE 'nova_labs%'")
        )
    ).one()
    assert str(row[0]) == "2026-09-02" and row[1] == "Whitefield" and row[2] == "news"


async def test_duplicate_document_is_skipped(db_session, indexer) -> None:
    await indexer.index_path(db_session, DOCS / "helix_flat.txt")
    again = await indexer.index_path(db_session, DOCS / "helix_flat.txt")
    assert again.documents_skipped_duplicate == 1 and again.documents_created == 0
    count = (
        await db_session.execute(text("SELECT count(*) FROM documents WHERE title LIKE 'helix%'"))
    ).scalar_one()
    assert count == 1


async def test_html_scripts_stripped_and_timestamp_date_parsed(db_session, indexer) -> None:
    await indexer.index_path(db_session, DOCS / "zephyr_funding.html", document_type="news")
    content = " ".join(r[0] for r in (await db_session.execute(text("SELECT content FROM document_chunks"))).all())
    assert "alert(" not in content and "Series A" in content and "Home | About" not in content
    published = (await db_session.execute(text("SELECT published_on FROM documents"))).scalar_one()
    assert str(published) == "2026-08-20"


async def test_unsupported_and_invalid_type_reported_not_raised(db_session, indexer, tmp_path) -> None:
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    assert (await indexer.index_path(db_session, bad)).failures
    assert (await indexer.index_path(db_session, DOCS / "helix_flat.txt", document_type="bogus")).failures


async def test_malformed_pdf_reported_not_raised(db_session, indexer, tmp_path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    r = await indexer.index_path(db_session, bad)
    assert r.failures and r.documents_created == 0


async def test_embedding_reuses_identical_chunk_text(db_session, indexer) -> None:
    await indexer.index_path(db_session, DOCS / "helix_flat.txt")
    emb = FakeEmbedder(384)
    svc = EmbeddingService(emb, batch_size=8)
    s1 = await svc.embed_pending(db_session)
    assert s1.embedded >= 1 and s1.pending_after == 0
    await db_session.execute(
        text("INSERT INTO documents (title, document_type, content_hash) VALUES ('copy','other','copyhash')")
    )
    did = (await db_session.execute(text("SELECT id FROM documents WHERE content_hash='copyhash'"))).scalar_one()
    src = (await db_session.execute(text("SELECT content, content_hash FROM document_chunks LIMIT 1"))).one()
    await db_session.execute(
        text("INSERT INTO document_chunks (document_id, chunk_index, content, content_hash) VALUES (:d,0,:c,:h)"),
        {"d": did, "c": src[0], "h": src[1]},
    )
    before = emb.texts_embedded
    s2 = await svc.embed_pending(db_session)
    assert s2.reused_from_cache == 1 and emb.texts_embedded == before


async def test_provider_outage_leaves_chunks_pending_not_lost(db_session, indexer) -> None:
    class Down(FakeEmbedder):
        async def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            raise EmbeddingUnavailableError("down")

    await indexer.index_path(db_session, DOCS / "nova_labs_expansion.txt")
    stats = await EmbeddingService(Down(384)).embed_pending(db_session)
    assert stats.failed_batches >= 1 and stats.embedded == 0 and stats.pending_after >= 1
    pending = (
        await db_session.execute(text("SELECT count(*) FROM document_chunks WHERE embedding IS NULL"))
    ).scalar_one()
    assert pending >= 1