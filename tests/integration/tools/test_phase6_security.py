from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings
from app.tools.db.readonly import ReadOnlyDatabase


@pytest.mark.asyncio
async def test_agent_ro_can_read_approved_views():
    settings = Settings()
    db = ReadOnlyDatabase(settings)

    try:
        expected_views = (
            "v_companies",
            "v_company_locations",
            "v_company_hiring",
            "v_company_funding",
            "v_coworking_branches",
            "v_coworking_tenants",
            "v_properties",
            "v_signals",
            "v_signal_scores",
        )

        for view_name in expected_views:
            count = await db.scalar(
                f"SELECT COUNT(*) FROM {view_name}"
            )
            assert count is not None
            assert int(count) >= 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_agent_ro_cannot_execute_write_sql():
    settings = Settings()

    engine = create_async_engine(
        str(settings.agent_ro_database_url),
        execution_options={
            "isolation_level": "AUTOCOMMIT",
        },
    )

    try:
        with pytest.raises(Exception):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        """
                        CREATE TABLE phase6_security_test (
                            id integer
                        )
                        """
                    )
                )
    finally:
        await engine.dispose()