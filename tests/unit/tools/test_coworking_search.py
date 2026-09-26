from unittest.mock import AsyncMock

import pytest

from app.tools.coworking_search import coworking_search
from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import CoworkingSearchInput


@pytest.mark.asyncio
async def test_coworking_search_filters() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)

    db.fetch_all.return_value = [
        {
            "branch_id": "branch-1",
            "branch_name": "Galaxy",
            "seat_capacity": 500,
            "operator_name": "WeWork",
            "building_id": "building-1",
            "building_name": "Galaxy",
            "area": "Whitefield",
            "city": "Bengaluru",
            "latitude": 12.97,
            "longitude": 77.75,
        }
    ]

    result = await coworking_search(
        CoworkingSearchInput(
            operator_name="WeWork",
            city="Bengaluru",
            area="Whitefield",
            limit=5,
        ),
        db,
    )

    assert len(result) == 1
    assert result[0]["operator_name"] == "WeWork"

    sql, params = db.fetch_all.await_args.args

    assert "FROM v_coworking_branches" in sql
    assert params["operator_name"] == "%WeWork%"
    assert params["city"] == "%Bengaluru%"
    assert params["area"] == "%Whitefield%"
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_coworking_search_is_parameterized() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)
    db.fetch_all.return_value = []

    attack = "'; DROP TABLE companies; --"

    await coworking_search(
        CoworkingSearchInput(operator_name=attack),
        db,
    )

    sql, params = db.fetch_all.await_args.args

    assert attack not in sql
    assert params["operator_name"] == f"%{attack}%"