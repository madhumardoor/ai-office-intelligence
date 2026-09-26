from __future__ import annotations

from typing import Any

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import CompanySearchInput


async def company_search(
    request: CompanySearchInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    """Search companies using the PII-free v_companies view."""

    conditions: list[str] = []
    params: dict[str, Any] = {"limit": request.limit}

    if request.query:
        conditions.append(
            """
            (
                name ILIKE :query
                OR domain ILIKE :query
                OR website ILIKE :query
            )
            """
        )
        params["query"] = f"%{request.query}%"

    if request.city:
        conditions.append("city ILIKE :city")
        params["city"] = f"%{request.city}%"

    if request.industry:
        conditions.append("industry ILIKE :industry")
        params["industry"] = f"%{request.industry}%"

    if request.min_employee_count is not None:
        conditions.append("employee_count >= :min_employee_count")
        params["min_employee_count"] = request.min_employee_count

    if request.max_employee_count is not None:
        conditions.append("employee_count <= :max_employee_count")
        params["max_employee_count"] = request.max_employee_count

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(
            f"({condition.strip()})"
            for condition in conditions
        )

    sql = f"""
        SELECT
            id,
            name,
            domain,
            website,
            industry,
            employee_count,
            city,
            state
        FROM v_companies
        {where_clause}
        ORDER BY
            CASE
                WHEN :query_exact IS NOT NULL
                     AND lower(name) = lower(:query_exact)
                THEN 0
                ELSE 1
            END,
            name
        LIMIT :limit
    """

    params["query_exact"] = request.query

    return await db.fetch_all(sql, params)