import pytest

from app.tools.db import ReadOnlyDatabase, ToolDatabaseError


pytestmark = pytest.mark.integration


async def test_agent_ro_can_read() -> None:
    db = ReadOnlyDatabase()

    try:
        rows = await db.fetch_all(
            """
            SELECT name, city
            FROM v_companies
            ORDER BY name
            LIMIT :limit
            """,
            {"limit": 3},
        )

        assert len(rows) <= 3
        assert all("name" in row for row in rows)

    finally:
        await db.close()


async def test_non_select_is_rejected_before_execution() -> None:
    db = ReadOnlyDatabase()

    try:
        with pytest.raises(
            ToolDatabaseError,
            match="Only SELECT/CTE read queries are allowed",
        ):
            await db.fetch_all(
                "DELETE FROM companies WHERE 1 = 0"
            )

    finally:
        await db.close()


async def test_multiple_statements_are_rejected() -> None:
    db = ReadOnlyDatabase()

    try:
        with pytest.raises(
            ToolDatabaseError,
            match="Multiple SQL statements are not allowed",
        ):
            await db.fetch_all(
                "SELECT 1; SELECT 2"
            )

    finally:
        await db.close()