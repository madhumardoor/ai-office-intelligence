from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.tools.schemas.inputs import SignalLookupInput
from app.tools.signal_lookup import signal_lookup


@pytest.mark.asyncio
async def test_signal_lookup_uses_parameterized_filters():
    company_id = uuid4()

    db = AsyncMock()
    db.fetch_all.return_value = []

    request = SignalLookupInput(
        company_id=company_id,
        signal_code="expansion",
        kind="signal",
        city="Bangalore",
        limit=10,
    )

    result = await signal_lookup(request, db)

    assert result == []

    sql, params = db.fetch_all.await_args.args

    assert "FROM v_signals s" in sql
    assert "LEFT JOIN v_companies c" in sql
    assert "s.company_id = :company_id" in sql
    assert "s.signal_code = :signal_code" in sql
    assert "s.kind = :kind" in sql
    assert "c.city = :city" in sql
    assert "LIMIT :limit" in sql

    assert params["company_id"] == company_id
    assert params["signal_code"] == "expansion"
    assert params["kind"] == "signal"
    assert params["city"] == "Bangalore"
    assert params["limit"] == 10

    assert "expansion" not in sql
    assert "Bangalore" not in sql


@pytest.mark.asyncio
async def test_signal_lookup_maps_rows_without_building_sql_from_values():
    signal_id = uuid4()
    company_id = uuid4()

    db = AsyncMock()
    db.fetch_all.return_value = [
        {
            "id": signal_id,
            "company_id": company_id,
            "signal_code": "hiring",
            "kind": "signal",
            "evidence_text": "Hiring increased in Whitefield",
            "evidence_url": "https://example.com/source",
            "detected_on": date(2026, 9, 25),
            "value": 25.0,
            "confidence": "high",
            "source_name": "example",
        }
    ]

    request = SignalLookupInput(
        company_id=company_id,
        limit=5,
    )

    result = await signal_lookup(request, db)

    assert result == [
        {
            "id": str(signal_id),
            "company_id": str(company_id),
            "signal_code": "hiring",
            "kind": "signal",
            "evidence_text": "Hiring increased in Whitefield",
            "evidence_url": "https://example.com/source",
            "detected_on": "2026-09-25",
            "value": 25.0,
            "confidence": "high",
            "source_name": "example",
        }
    ]

    sql, params = db.fetch_all.await_args.args

    assert "hiring" not in sql
    assert "Whitefield" not in sql
    assert params["company_id"] == company_id