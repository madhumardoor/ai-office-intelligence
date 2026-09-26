"""Database health probes used by /health/ready."""

from __future__ import annotations

import logging
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.schemas.health import ComponentStatus

logger = logging.getLogger(__name__)

REQUIRED_EXTENSIONS = frozenset({"postgis", "vector", "pg_trgm", "pgcrypto"})


async def check_database(engine: AsyncEngine) -> ComponentStatus:
    """Verify connectivity and measure round-trip latency."""
    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ComponentStatus(name="postgres", ok=True, latency_ms=round(latency, 2))
    except Exception as exc:  # noqa: BLE001 - health probes must never raise
        logger.error("database health check failed", extra={"error": str(exc)})
        return ComponentStatus(name="postgres", ok=False, detail="connection failed")


async def check_extensions(engine: AsyncEngine) -> ComponentStatus:
    """Verify that all required Postgres extensions are installed."""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT extname FROM pg_extension"))
            installed = {row[0] for row in result}
        missing = REQUIRED_EXTENSIONS - installed
        if missing:
            return ComponentStatus(
                name="extensions", ok=False, detail=f"missing: {', '.join(sorted(missing))}"
            )
        return ComponentStatus(name="extensions", ok=True, detail=", ".join(sorted(installed & REQUIRED_EXTENSIONS)))
    except Exception as exc:  # noqa: BLE001
        logger.error("extension health check failed", extra={"error": str(exc)})
        return ComponentStatus(name="extensions", ok=False, detail="check failed")