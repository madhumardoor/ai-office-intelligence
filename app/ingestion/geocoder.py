"""Cached, rate-limited geocoding via Nominatim. Only called when lat/lng is missing AND enabled."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.models.ops import GeocodeCache

logger = logging.getLogger(__name__)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@dataclass(frozen=True, slots=True)
class GeoResult:
    latitude: float
    longitude: float


class Geocoder:
    """Nominatim policy: <= 1 request/second, identifying User-Agent. Both are enforced here."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._enabled = settings.geocoding_enabled
        self._min_interval = settings.geocoding_min_interval_seconds
        self._headers = {"User-Agent": settings.geocoding_user_agent}
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    async def geocode(self, session: AsyncSession, query: str) -> GeoResult | None:
        if not self._enabled or not query.strip():
            return None
        key = self._key(query)
        cached = (
            await session.execute(select(GeocodeCache).where(GeocodeCache.query_hash == key))
        ).scalar_one_or_none()
        if cached is not None:  # hit (including cached misses)
            if cached.found and cached.latitude is not None and cached.longitude is not None:
                return GeoResult(cached.latitude, cached.longitude)
            return None

        try:
            result = await self._fetch(query)
        except Exception as exc:  # noqa: BLE001 - do NOT cache transient failures
            logger.warning("geocoding failed", extra={"error": type(exc).__name__})
            return None

        stmt = (
            insert(GeocodeCache)
            .values(
                query=query,
                query_hash=key,
                found=result is not None,
                latitude=result.latitude if result else None,
                longitude=result.longitude if result else None,
            )
            .on_conflict_do_nothing(index_elements=["query_hash"])
        )
        await session.execute(stmt)
        return result

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        reraise=True,
    )
    async def _fetch(self, query: str) -> GeoResult | None:
        async with self._lock:  # serialize + space out calls
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
        resp = await self._client.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers=self._headers,
        )
        resp.raise_for_status()  # 429/5xx raise and trigger retry with backoff
        data = resp.json()
        if not data:
            return None
        return GeoResult(float(data[0]["lat"]), float(data[0]["lon"]))