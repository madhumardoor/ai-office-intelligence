"""Global exception handlers: consistent JSON errors, no internal leakage."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.logging import request_id_ctx

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, details: object | None = None) -> JSONResponse:
    body: dict[str, object] = {
        "error": {"code": code, "message": message, "request_id": request_id_ctx.get()}
    }
    if details is not None:
        body["error"]["details"] = details  # type: ignore[index]
    return JSONResponse(status_code=status, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exc(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exc(_: Request, exc: RequestValidationError) -> JSONResponse:
        # jsonable-safe subset only: never echo raw input back (may contain secrets).
        safe = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return _error(422, "validation_error", "Request validation failed", safe)

    @app.exception_handler(Exception)
    async def unhandled_exc(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception", extra={"error_type": type(exc).__name__})
        return _error(500, "internal_error", "An unexpected error occurred")