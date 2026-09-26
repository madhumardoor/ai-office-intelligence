"""Verifies the agent role can read ONLY allow-listed views, and can never write.
Requires the agent_ro role, so this test creates it in the throwaway DB the same way 02-roles.sh does."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.integration


@pytest.fixture
async def ro_engine(migrated_db):
    admin = create_async_engine(str(migrated_db["url"]), isolation_level="AUTOCOMMIT")
    async with admin.connect() as c:
        await c.execute(text("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='agent_ro') "
                             "THEN CREATE ROLE agent_ro LOGIN PASSWORD 'ro-test'; END IF; END $$"))
        await c.execute(text("ALTER ROLE agent_ro SET default_transaction_read_only = on"))
        await c.execute(text("GRANT CONNECT ON DATABASE aoi TO agent_ro"))
        await c.execute(text("GRANT USAGE ON SCHEMA public TO agent_ro"))
        # Re-apply exactly the grants the migration applies (role did not exist when it ran).
        await c.execute(text("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_ro"))
        for v in ("v_companies", "v_coworking_tenants", "v_signals"):
            await c.execute(text(f"GRANT SELECT ON {v} TO agent_ro"))
        await c.execute(text("INSERT INTO companies (name, normalized_name, domain) "
                             "VALUES ('ROCo','roco','roco.example') ON CONFLICT DO NOTHING"))
    await admin.dispose()
    url = str(migrated_db["url"]).replace("aoi_admin:test-pass", "agent_ro:ro-test")
    engine = create_async_engine(url)
    yield engine
    await engine.dispose()


async def test_can_read_allowlisted_view(ro_engine) -> None:
    async with ro_engine.connect() as c:
        rows = (await c.execute(text("SELECT name FROM v_companies WHERE name='ROCo'"))).all()
    assert rows


async def test_cannot_read_base_tables(ro_engine) -> None:
    async with ro_engine.connect() as c:
        with pytest.raises(DBAPIError, match="permission denied"):
            await c.execute(text("SELECT * FROM companies"))


async def test_cannot_read_pii_tables_or_columns(ro_engine) -> None:
    async with ro_engine.connect() as c:
        with pytest.raises(DBAPIError, match="permission denied"):
            await c.execute(text("SELECT contact_email FROM coworking_tenants"))


@pytest.mark.parametrize("sql", [
    "INSERT INTO v_companies (name) VALUES ('x')",
    "UPDATE v_companies SET name='x'",
    "DELETE FROM v_companies",
    "DROP TABLE companies",
    "CREATE TABLE evil (id int)",
])
async def test_all_writes_rejected(ro_engine, sql: str) -> None:
    async with ro_engine.connect() as c:
        with pytest.raises(DBAPIError):
            await c.execute(text(sql))