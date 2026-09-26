from unittest.mock import AsyncMock

import pytest

from app.tools.db import ReadOnlyDatabase
from app.tools.postgis_search import postgis_search
from app.tools.schemas import PostGISSearchInput


@pytest.mark.asyncio
async def test_postgis_search_uses_parameterized_radius() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)

    db.fetch_all.return_value = [
        {
            "id": "location-1",
            "company_id": "company-1",
            "label": "Main Office",
            "area": "Whitefield",
            "city": "Bengaluru",
            "latitude": 12.97,
            "longitude": 77.75,
            "distance_km": 1.234,
        }
    ]

    result = await postgis_search(
        PostGISSearchInput(
            latitude=12.9716,
            longitude=77.5946,
            radius_km=10,
            city="Bengaluru",
            limit=5,
        ),
        db,
    )

    assert len(result) == 1
    assert result[0]["distance_km"] == 1.234

    sql, params = db.fetch_all.await_args.args

    assert "ST_DWithin" in sql
    assert "ST_Distance" in sql
    assert params["latitude"] == 12.9716
    assert params["longitude"] == 77.5946
    assert params["radius_m"] == 10000
    assert params["city"] == "%Bengaluru%"
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_postgis_search_does_not_inject_coordinates_into_sql() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)
    db.fetch_all.return_value = []

    await postgis_search(
        PostGISSearchInput(
            latitude=12.9716,
            longitude=77.5946,
            radius_km=5,
        ),
        db,
    )

    sql, params = db.fetch_all.await_args.args

    assert "12.9716" not in sql
    assert "77.5946" not in sql
    assert params["latitude"] == 12.9716
    assert params["longitude"] == 77.5946