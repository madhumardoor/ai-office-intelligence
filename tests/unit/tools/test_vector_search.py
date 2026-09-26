from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.tools.schemas import VectorSearchInput
from app.tools.vector_search import vector_search


@pytest.mark.asyncio
async def test_vector_search_uses_bounded_retrieval() -> None:
    company_id = uuid4()
    chunk_id = uuid4()
    document_id = uuid4()

    retriever = AsyncMock()

    retriever.retrieve.return_value = SimpleNamespace(
        chunks=[
            SimpleNamespace(
                chunk_id=chunk_id,
                document_id=document_id,
                company_id=company_id,
                title="Nova expansion",
                content="Nova Labs is expanding in Whitefield.",
                source_url="https://example.com/nova",
                document_type="news",
                published_on=date(2026, 9, 1),
                score=0.91,
                vector_rank=1,
                keyword_rank=2,
            )
        ]
    )

    result = await vector_search(
        VectorSearchInput(
            query="Nova Whitefield expansion",
            company_id=company_id,
            document_type="news",
            limit=5,
        ),
        retriever,
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == str(chunk_id)
    assert result[0]["company_id"] == str(company_id)
    assert result[0]["score"] == 0.91

    retriever.retrieve.assert_awaited_once()

    query = retriever.retrieve.await_args.args[0]
    kwargs = retriever.retrieve.await_args.kwargs
    filters = kwargs["filters"]

    assert query == "Nova Whitefield expansion"
    assert filters.company_ids == (company_id,)
    assert filters.document_types == ("news",)
    assert kwargs["top_k"] == 5