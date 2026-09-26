from __future__ import annotations

from typing import Any

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import PropertySearchInput


async def property_search(
    request: PropertySearchInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    """Search properties from the PII-free v_properties view."""

    conditions: list[str] = []
    params: dict[str, Any] = {"limit": request.limit}

    if request.query:
        conditions.append(
            """
            (
                name ILIKE :query
                OR property_type ILIKE :query
                OR area ILIKE :query
                OR city ILIKE :query
            )
            """
        )
        params["query"] = f"%{request.query}%"

    if request.property_type:
        conditions.append("property_type ILIKE :property_type")
        params["property_type"] = f"%{request.property_type}%"

    if request.city:
        conditions.append("city ILIKE :city")
        params["city"] = f"%{request.city}%"

    if request.area:
        conditions.append("area ILIKE :area")
        params["area"] = f"%{request.area}%"

    if request.min_area_sqft is not None:
        conditions.append("total_area_sqft >= :min_area_sqft")
        params["min_area_sqft"] = request.min_area_sqft

    if request.max_area_sqft is not None:
        conditions.append("total_area_sqft <= :max_area_sqft")
        params["max_area_sqft"] = request.max_area_sqft

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            id,
            name,
            property_type,
            area,
            city,
            latitude,
            longitude,
            total_area_sqft
        FROM v_properties
        {where_clause}
        ORDER BY name
        LIMIT :limit
    """

    return await db.fetch_all(sql, params)