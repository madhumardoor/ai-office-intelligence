"""Parameterised PostGIS queries. All inputs are bound parameters (no string interpolation).

Why geography instead of raw lat/lng:
  * ST_DWithin on geography takes METERS and uses the GiST index; a Haversine expression cannot.
  * Polygons (e.g. a Whitefield boundary) and KNN ordering (<->) come for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MAX_RADIUS_M = 100_000
MAX_LIMIT = 200


def _clamp(radius_km: float, limit: int) -> tuple[float, int]:
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    return min(radius_km * 1000.0, MAX_RADIUS_M), max(1, min(limit, MAX_LIMIT))


@dataclass(frozen=True, slots=True)
class NearbyCompany:
    company_id: UUID
    name: str
    area: str | None
    distance_m: float


async def companies_near_point(
    session: AsyncSession, *, lat: float, lng: float, radius_km: float, limit: int = 50
) -> list[NearbyCompany]:
    """Companies with any location within radius_km of a point, nearest first."""
    radius_m, limit = _clamp(radius_km, limit)
    sql = text(
        """
        SELECT DISTINCT ON (c.id) c.id, c.name, cl.area,
               ST_Distance(cl.geog, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography) AS dist
        FROM company_locations cl
        JOIN companies c ON c.id = cl.company_id AND c.deleted_at IS NULL
        WHERE ST_DWithin(cl.geog, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography, :radius_m)
        ORDER BY c.id, dist
        """
    )
    rows = (await session.execute(sql, {"lat": lat, "lng": lng, "radius_m": radius_m})).all()
    result = sorted(
        (NearbyCompany(r.id, r.name, r.area, float(r.dist)) for r in rows), key=lambda x: x.distance_m
    )
    return result[:limit]


async def companies_near_building(
    session: AsyncSession, *, building_id: UUID, radius_km: float, limit: int = 50
) -> list[NearbyCompany]:
    """Companies within radius_km of a coworking building."""
    radius_m, limit = _clamp(radius_km, limit)
    sql = text(
        """
        SELECT DISTINCT ON (c.id) c.id, c.name, cl.area,
               ST_Distance(cl.geog, b.geog) AS dist
        FROM coworking_buildings b
        JOIN company_locations cl ON ST_DWithin(cl.geog, b.geog, :radius_m)
        JOIN companies c ON c.id = cl.company_id AND c.deleted_at IS NULL
        WHERE b.id = :building_id
        ORDER BY c.id, dist
        """
    )
    rows = (await session.execute(sql, {"building_id": building_id, "radius_m": radius_m})).all()
    result = sorted(
        (NearbyCompany(r.id, r.name, r.area, float(r.dist)) for r in rows), key=lambda x: x.distance_m
    )
    return result[:limit]


async def nearest_coworking_buildings(
    session: AsyncSession, *, lat: float, lng: float, limit: int = 5
) -> list[dict[str, Any]]:
    """K-nearest coworking buildings using the GiST-assisted <-> operator."""
    _, limit = _clamp(1, limit)
    sql = text(
        """
        SELECT b.id, b.name, b.area,
               ST_Distance(b.geog, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography) AS dist
        FROM coworking_buildings b
        ORDER BY b.geog <-> ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography
        LIMIT :limit
        """
    )
    rows = (await session.execute(sql, {"lat": lat, "lng": lng, "limit": limit})).all()
    return [{"id": r.id, "name": r.name, "area": r.area, "distance_m": float(r.dist)} for r in rows]