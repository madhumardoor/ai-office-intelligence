from unittest.mock import AsyncMock

import pytest

from app.tools.company_search import company_search
from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import CompanySearchInput


@pytest.mark.asyncio
async def test_company_search_passes_parameterized_filters() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)

    db.fetch_all.return_value = [
        {
            "id": "company-1",
            "name": "Nova Labs",
            "domain": "nova.example",
            "website": "https://nova.example",
            "industry": "Software",
            "employee_count": 120,
            "city": "Bengaluru",
            "state": "Karnataka",
        }
    ]

    result = await company_search(
        CompanySearchInput(
            query="Nova",
            city="Bengaluru",
            industry="Software",
            min_employee_count=50,
            max_employee_count=500,
            limit=5,
        ),
        db,
    )

    assert len(result) == 1
    db.fetch_all.assert_awaited_once()

    sql, params = db.fetch_all.await_args.args

    assert "FROM v_companies" in sql
    assert "ILIKE :query" in sql
    assert params["query"] == "%Nova%"
    assert params["city"] == "%Bengaluru%"
    assert params["industry"] == "%Software%"
    assert params["min_employee_count"] == 50
    assert params["max_employee_count"] == 500
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_company_search_does_not_insert_user_text_into_sql() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)
    db.fetch_all.return_value = []

    malicious = "'; DROP TABLE companies; --"

    await company_search(
        CompanySearchInput(query=malicious),
        db,
    )

    sql, params = db.fetch_all.await_args.args

    assert malicious not in sql
    assert params["query"] == f"%{malicious}%"
