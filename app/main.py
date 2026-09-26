"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.routers import health
from app.config import Settings, get_settings
from app.config.logging import configure_logging
from app.database import create_engine, create_session_factory

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (enables per-test settings without global state)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        logger.info("application started", extra={"env": settings.app_env})
        try:
            yield
        finally:
            await engine.dispose()
            logger.info("application stopped")

    app = FastAPI(
        title="AI Office Demand Intelligence Platform",
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs off in production; enable behind auth later if needed.
        docs_url=None if settings.app_env == "production" else "/docs",
        redoc_url=None,
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    return app


app = create_app()