from unittest.mock import AsyncMock

import pytest

from app.tools.db import ReadOnlyDatabase
from app.tools.property_search import property_search
from app.tools.schemas import PropertySearchInput


@pytest.mark.asyncio
async def test_property_search_filters() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)

    db.fetch_all.return_value = [
        {
            "id": "property-1",
            "name": "Tech Park",
            "property_type": "Commercial",
            "area": "Whitefield",
            "city": "Bengaluru",
            "latitude": 12.97,
            "longitude": 77.75,
            "total_area_sqft": 100000,
        }
    ]

    result = await property_search(
        PropertySearchInput(
            query="Tech",
            property_type="Commercial",
            city="Bengaluru",
            area="Whitefield",
            min_area_sqft=50000,
            max_area_sqft=150000,
            limit=5,
        ),
        db,
    )

    assert len(result) == 1
    assert result[0]["name"] == "Tech Park"

    sql, params = db.fetch_all.await_args.args

    assert "FROM v_properties" in sql
    assert params["query"] == "%Tech%"
    assert params["property_type"] == "%Commercial%"
    assert params["city"] == "%Bengaluru%"
    assert params["area"] == "%Whitefield%"
    assert params["min_area_sqft"] == 50000
    assert params["max_area_sqft"] == 150000
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_property_search_is_parameterized() -> None:
    db = AsyncMock(spec=ReadOnlyDatabase)
    db.fetch_all.return_value = []

    attack = "'; DROP TABLE properties; --"

    await property_search(
        PropertySearchInput(query=attack),
        db,
    )

    sql, params = db.fetch_all.await_args.args

    assert attack not in sql
    assert params["query"] == f"%{attack}%"