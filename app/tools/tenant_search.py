from __future__ import annotations

from typing import Any

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import TenantSearchInput


async def tenant_search(
    request: TenantSearchInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    """Search coworking tenants from the PII-free view."""

    conditions: list[str] = []
    params: dict[str, Any] = {"limit": request.limit}

    if request.query:
        conditions.append("company_name ILIKE :query")
        params["query"] = f"%{request.query}%"

    if request.company_id is not None:
        conditions.append("company_id = :company_id")
        params["company_id"] = request.company_id

    if request.branch_id is not None:
        conditions.append("branch_id = :branch_id")
        params["branch_id"] = request.branch_id

    if request.min_seats_used is not None:
        conditions.append("seats_used >= :min_seats_used")
        params["min_seats_used"] = request.min_seats_used

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            id,
            company_id,
            company_name,
            branch_id,
            seats_used,
            first_seen_on,
            last_seen_on
        FROM v_coworking_tenants
        {where_clause}
        ORDER BY company_name, last_seen_on DESC
        LIMIT :limit
    """

    return await db.fetch_all(sql, params)