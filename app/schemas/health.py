"""Health-check response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    name: str
    ok: bool
    latency_ms: float | None = None
    detail: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    components: list[ComponentStatus]