from __future__ import annotations

from typing import Any

from app.tools.db.readonly import ReadOnlyDatabase
from app.tools.schemas.inputs import SignalLookupInput


async def signal_lookup(
    request: SignalLookupInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "limit": request.limit,
    }

    conditions: list[str] = []

    if request.company_id is not None:
        conditions.append("s.company_id = :company_id")
        params["company_id"] = request.company_id

    if request.signal_code is not None:
        conditions.append("s.signal_code = :signal_code")
        params["signal_code"] = request.signal_code

    if request.kind is not None:
        conditions.append("s.kind = :kind")
        params["kind"] = request.kind

    if request.city is not None:
        conditions.append("c.city = :city")
        params["city"] = request.city

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT
            s.id,
            s.company_id,
            s.signal_code,
            s.kind,
            s.evidence_text,
            s.evidence_url,
            s.detected_on,
            s.value,
            s.confidence,
            s.source_name
        FROM v_signals s
        LEFT JOIN v_companies c
            ON c.id = s.company_id
        {where_sql}
        ORDER BY s.detected_on DESC NULLS LAST
        LIMIT :limit
    """

    rows = await db.fetch_all(sql, params)

    return [
        {
            "id": str(row["id"]),
            "company_id": str(row["company_id"]),
            "signal_code": row["signal_code"],
            "kind": row["kind"],
            "evidence_text": row["evidence_text"],
            "evidence_url": row["evidence_url"],
            "detected_on": (
                row["detected_on"].isoformat()
                if row["detected_on"] is not None
                else None
            ),
            "value": row["value"],
            "confidence": row["confidence"],
            "source_name": row["source_name"],
        }
        for row in rows
    ]