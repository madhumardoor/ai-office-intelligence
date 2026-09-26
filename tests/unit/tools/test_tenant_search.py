from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import TenantSearchInput
from app.tools.tenant_search import tenant_search


@pytest.mark.asyncio
async def test_tenant_search_filters() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)

    company_id = uuid4()
    branch_id = uuid4()

    db.fetch_all.return_value = [
        {
            "id": "tenant-1",
            "company_id": company_id,
            "company_name": "Nova Labs",
            "branch_id": branch_id,
            "seats_used": 40,
            "first_seen_on": None,
            "last_seen_on": None,
        }
    ]

    result = await tenant_search(
        TenantSearchInput(
            query="Nova",
            company_id=company_id,
            branch_id=branch_id,
            min_seats_used=20,
            limit=5,
        ),
        db,
    )

    assert len(result) == 1
    assert result[0]["company_name"] == "Nova Labs"

    sql, params = db.fetch_all.await_args.args

    assert "FROM v_coworking_tenants" in sql
    assert params["query"] == "%Nova%"
    assert params["company_id"] == company_id
    assert params["branch_id"] == branch_id
    assert params["min_seats_used"] == 20
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_tenant_search_is_parameterized() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)
    db.fetch_all.return_value = []

    attack = "'; DROP TABLE companies; --"

    await tenant_search(
        TenantSearchInput(query=attack),
        db,
    )

    sql, params = db.fetch_all.await_args.args

    assert attack not in sql
    assert params["query"] == f"%{attack}%"