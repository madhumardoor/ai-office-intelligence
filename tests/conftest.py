"""Shared fixtures.

Integration tests spin up a REAL PostGIS + pgvector container.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from testcontainers.community.postgres import PostgresContainer

from app.config import Settings
from app.main import create_app


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_settings() -> Settings:
    """Application settings used by integration tests."""
    return Settings(
        app_env="test",
        log_json=True,
        postgres_user="aoi_admin",
        postgres_password=SecretStr("test-pass"),
        postgres_db="aoi",
        agent_ro_password=SecretStr("test-ro-pass"),
    )


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    """Start a real PostgreSQL container with PostGIS and pgvector.

    The image is built from the project's Dockerfile:

        docker/postgres.Dockerfile

    Requires Docker to be installed and running.

    Set SKIP_INTEGRATION=1 to explicitly skip integration tests.
    """

    if os.environ.get("SKIP_INTEGRATION") == "1":
        pytest.skip("SKIP_INTEGRATION=1")

    try:
        from testcontainers.core.image import DockerImage

        # Build the custom PostgreSQL image.
        image = DockerImage(
            path=str(REPO_ROOT),
            dockerfile_path="docker/postgres.Dockerfile",
            tag="aoi-pg-test:latest",
        )

        image.build()

        # Start PostgreSQL using the community testcontainers package.
        container = PostgresContainer(
            image="aoi-pg-test:latest",
            username="aoi_admin",
            password="test-pass",
            dbname="aoi",
            driver=None,
        )

        container.start()

    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Docker/testcontainers unavailable: {exc}"
        )

    try:
        yield container

    finally:
        container.stop()


@pytest.fixture
async def integration_client(
    pg_container: PostgresContainer,
    test_settings: Settings,
) -> AsyncIterator[AsyncClient]:
    """Create an HTTP client connected to the temporary test database."""

    from sqlalchemy import text

    from app.database import create_engine

    settings = test_settings.model_copy(
        update={
            "postgres_host": pg_container.get_container_host_ip(),
            "postgres_port": int(
                pg_container.get_exposed_port(5432)
            ),
        }
    )

    engine = create_engine(settings)

    # Ensure required PostgreSQL extensions exist.
    async with engine.begin() as conn:
        for extension in (
            "postgis",
            "vector",
            "pg_trgm",
            "pgcrypto",
        ):
            await conn.execute(
                text(
                    f"CREATE EXTENSION IF NOT EXISTS {extension}"
                )
            )

    await engine.dispose()

    # Create the application using the temporary database settings.
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client


@pytest.fixture(scope="session")
def migrated_db(
    pg_container: PostgresContainer,
) -> dict[str, object]:
    """Run the real Alembic migrations against the test container."""

    host = pg_container.get_container_host_ip()

    port = int(
        pg_container.get_exposed_port(5432)
    )

    database_url = (
        "postgresql+asyncpg://"
        f"aoi_admin:test-pass@{host}:{port}/aoi"
    )

    env = {
        **os.environ,
        "ALEMBIC_DATABASE_URL": database_url,
    }

    # Run all Alembic migrations.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            "head",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    return {
        "url": database_url,
        "host": host,
        "port": port,
    }


@pytest.fixture
async def db_session(
    migrated_db: dict[str, object],
):
    """Provide an isolated database session for each integration test.

    A top-level transaction is rolled back after the test.

    A nested SAVEPOINT allows application code to safely flush/commit
    while the outer transaction maintains test isolation.
    """

    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        create_async_engine,
    )

    engine = create_async_engine(
        str(migrated_db["url"])
    )

    async with engine.connect() as conn:
        # Start the outer transaction.
        outer_transaction = await conn.begin()

        session = AsyncSession(
            bind=conn,
            expire_on_commit=False,
        )

        # Start a nested SAVEPOINT.
        await session.begin_nested()

        @event.listens_for(
            session.sync_session,
            "after_transaction_end",
        )
        def restart_savepoint(
            session_sync,
            transaction,
        ):
            """Restart the SAVEPOINT after application commits."""

            if (
                transaction.nested
                and transaction._parent is not None
                and not transaction._parent.nested
            ):
                session_sync.begin_nested()

        try:
            yield session

        finally:
            await session.close()

            # Roll back the outer transaction to isolate the test.
            if outer_transaction.is_active:
                await outer_transaction.rollback()

    await engine.dispose()