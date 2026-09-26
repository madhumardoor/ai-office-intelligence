"""Request-ID and request-logging middleware (pure ASGI, no BaseHTTPMiddleware)."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config.logging import request_id_ctx

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = b"x-request-id"
# Only accept well-formed inbound IDs; otherwise a caller could inject log content.
_VALID_ID = re.compile(r"^[A-Za-z0-9\-_.]{8,64}$")


class RequestIdMiddleware:
    """Attach a request ID to every request/response and log request completion.

    Pure ASGI implementation: avoids BaseHTTPMiddleware's known issues with
    contextvars propagation and streaming responses (relevant for LLM streaming later).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = dict(scope["headers"]).get(REQUEST_ID_HEADER, b"").decode()
        request_id = inbound if _VALID_ID.match(inbound) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request completed",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            request_id_ctx.reset(token)