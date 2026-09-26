"""Hybrid retrieval end to end with the FAKE embedder (tests mechanics: SQL, filters, RRF, degradation).
Real-model retrieval quality is checked separately by the search CLI and a later evaluation phase.
"""

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.rag.embedding.base import EmbeddingUnavailableError
from app.rag.embedding.fake import FakeEmbedder
from app.rag.embedding.service import EmbeddingService
from app.rag.indexer import DocumentIndexer
from app.retrieval.context import build_context
from app.retrieval.filters import RetrievalFilters
from app.retrieval.hybrid import HybridRetriever


pytestmark = pytest.mark.integration

DOCS = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "samples"
    / "docs"
)


@pytest.fixture
async def stack(migrated_db):
    """Create an isolated hybrid-retrieval test dataset."""

    engine = create_async_engine(
        str(migrated_db["url"])
    )

    sf = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    # Start each hybrid test with a clean database.
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE "
                "document_chunks, "
                "documents, "
                "companies, "
                "sources "
                "CASCADE"
            )
        )

    emb = FakeEmbedder(384)

    async with sf() as session, session.begin():
        company_id = (
            await session.execute(
                text(
                    "INSERT INTO companies "
                    "(name, normalized_name, domain) "
                    "VALUES "
                    "('Nova Labs Pvt Ltd', "
                    "'nova labs', "
                    "'nova.example') "
                    "RETURNING id"
                )
            )
        ).scalar_one()

        indexer = DocumentIndexer(
            Settings(app_env="test")
        )

        await indexer.index_path(
            session,
            DOCS / "nova_labs_expansion.txt",
            document_type="news",
            company_id=company_id,
        )

        await indexer.index_path(
            session,
            DOCS / "zephyr_funding.html",
            document_type="funding_announcement",
        )

        await indexer.index_path(
            session,
            DOCS / "helix_flat.txt",
            document_type="report",
        )

        await EmbeddingService(emb).embed_pending(
            session
        )

    yield (
        HybridRetriever(sf, emb),
        emb,
        sf,
        company_id,
    )

    # Clean up everything created by this fixture.
    # This prevents hybrid tests from contaminating
    # later integration tests.
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE "
                "document_chunks, "
                "documents, "
                "companies, "
                "sources "
                "CASCADE"
            )
        )

    await engine.dispose()


async def test_relevant_document_ranked_first(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "Nova Labs hiring engineers Whitefield expansion",
        top_k=3,
    )

    assert result.chunks
    assert "Nova Labs" in result.chunks[0].content

    assert set(result.methods_used) == {
        "keyword",
        "vector",
    }

    assert not result.degraded


async def test_both_rank_signals_recorded(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "Series A funding Zephyr",
        top_k=3,
    )

    top = result.chunks[0]

    assert top.vector_rank is not None
    assert top.keyword_rank is not None


async def test_document_type_filter_excludes_others(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "headcount office",
        RetrievalFilters(
            document_types=("report",)
        ),
        top_k=5,
    )

    assert result.chunks

    assert all(
        chunk.document_type == "report"
        for chunk in result.chunks
    )


async def test_company_filter(
    stack,
) -> None:
    retriever, _, _, company_id = stack

    result = await retriever.retrieve(
        "expansion hiring funding",
        RetrievalFilters(
            company_ids=(company_id,)
        ),
        top_k=5,
    )

    assert result.chunks

    assert all(
        chunk.company_id == company_id
        for chunk in result.chunks
    )


async def test_date_filter(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "headcount",
        RetrievalFilters(
            published_before=date(
                2025,
                12,
                31,
            )
        ),
        top_k=5,
    )

    assert result.chunks

    assert all(
        chunk.published_on
        and chunk.published_on
        <= date(
            2025,
            12,
            31,
        )
        for chunk in result.chunks
    )


async def test_no_match_returns_empty_not_error(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "zzzz qqqq xxxx",
        RetrievalFilters(
            document_types=("news",),
            published_after=date(
                2030,
                1,
                1,
            ),
        ),
        top_k=5,
    )

    assert result.chunks == []


async def test_embedding_outage_degrades_to_keyword_only(
    stack,
) -> None:
    _, _, session_factory, _ = stack

    class Down(FakeEmbedder):
        async def embed_query(
            self,
            text: str,
        ):  # type: ignore[no-untyped-def]
            raise EmbeddingUnavailableError(
                "down"
            )

    result = await HybridRetriever(
        session_factory,
        Down(384),
    ).retrieve(
        "Series A funding",
        top_k=3,
    )

    assert result.chunks

    assert result.methods_used == [
        "keyword"
    ]

    assert (
        "vector_unavailable"
        in result.degraded
    )


async def test_injection_text_is_retrievable_as_data_but_context_neutralised(
    stack,
) -> None:
    retriever, *_ = stack

    result = await retriever.retrieve(
        "Zephyr Systems office space",
        top_k=5,
    )

    context = build_context(
        result.chunks
    )

    assert (
        "definitely needs 10,000 sqft"
        in context.text
    )

    assert (
        context.text.count("</evidence>")
        == len(context.chunks)
    )