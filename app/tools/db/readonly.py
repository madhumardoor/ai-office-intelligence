from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings


class ToolDatabaseError(RuntimeError):
    """Raised when a Phase 6 database operation is invalid or fails."""


_SELECT_PATTERN = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)")
_MULTI_STATEMENT_PATTERN = re.compile(r";")


def _validate_read_query(sql: str) -> None:
    """
    Validate SQL before it reaches Postgres.

    Phase 6 tools must issue read-only SQL only.
    The actual DB role is also read-only, giving us defense in depth.
    """
    normalized = sql.strip()

    if not normalized:
        raise ToolDatabaseError("SQL statement cannot be empty")

    if not _SELECT_PATTERN.match(normalized):
        raise ToolDatabaseError("Only SELECT/CTE read queries are allowed")

    if _COMMENT_PATTERN.search(normalized):
        raise ToolDatabaseError("SQL comments are not allowed")

    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        raise ToolDatabaseError("Multiple SQL statements are not allowed")


class ReadOnlyDatabase:
    """
    Shared database access layer for Phase 6 tools.

    All connections use the agent_ro role.
    All callers receive rows as dictionaries.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._engine: AsyncEngine = create_async_engine(
            self._settings.agent_ro_database_url,
            pool_size=self._settings.db_pool_size,
            max_overflow=self._settings.db_max_overflow,
            pool_timeout=self._settings.db_pool_timeout_seconds,
            pool_pre_ping=True,
            connect_args={
                "server_settings": {
                    "default_transaction_read_only": "on",
                    "statement_timeout": str(
                        self._settings.db_statement_timeout_ms
                    ),
                    "application_name": "aoi-phase6-tools",
                }
            },
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    def session(self) -> AsyncSession:
        return self._session_factory()

    async def fetch_all(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute one parameterized SELECT/CTE query.
        """
        _validate_read_query(sql)

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text(sql),
                    dict(params or {}),
                )

                rows: Sequence[RowMapping] = result.mappings().all()

                return [dict(row) for row in rows]

        except ToolDatabaseError:
            raise

        except Exception as exc:
            raise ToolDatabaseError(
                f"Read-only database query failed: {type(exc).__name__}"
            ) from exc

    async def fetch_one(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Execute one parameterized SELECT/CTE query and return one row.
        """
        rows = await self.fetch_all(sql, params)
        return rows[0] if rows else None

    async def scalar(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        Execute a read query expected to return one scalar value.
        """
        _validate_read_query(sql)

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text(sql),
                    dict(params or {}),
                )
                return result.scalar_one_or_none()

        except ToolDatabaseError:
            raise

        except Exception as exc:
            raise ToolDatabaseError(
                f"Read-only scalar query failed: {type(exc).__name__}"
            ) from exc

    async def close(self) -> None:
        """Dispose the connection pool."""
        await self._engine.dispose()