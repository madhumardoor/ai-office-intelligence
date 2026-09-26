from __future__ import annotations

from typing import Any

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import CoworkingSearchInput


async def coworking_search(
    request: CoworkingSearchInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    """Search coworking branches from the PII-free view."""

    conditions: list[str] = []
    params: dict[str, Any] = {"limit": request.limit}

    if request.operator_name:
        conditions.append("operator_name ILIKE :operator_name")
        params["operator_name"] = f"%{request.operator_name}%"

    if request.city:
        conditions.append("city ILIKE :city")
        params["city"] = f"%{request.city}%"

    if request.area:
        conditions.append("area ILIKE :area")
        params["area"] = f"%{request.area}%"

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            branch_id,
            branch_name,
            seat_capacity,
            operator_name,
            building_id,
            building_name,
            area,
            city,
            latitude,
            longitude
        FROM v_coworking_branches
        {where_clause}
        ORDER BY operator_name, branch_name
        LIMIT :limit
    """

    return await db.fetch_all(sql, params)