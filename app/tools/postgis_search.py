from __future__ import annotations

from typing import Any

from app.tools.db import ReadOnlyDatabase
from app.tools.schemas import PostGISSearchInput


async def postgis_search(
    request: PostGISSearchInput,
    db: ReadOnlyDatabase,
) -> list[dict[str, Any]]:
    """
    Find company locations within a radius using PostGIS.

    Distance is calculated in kilometers from the supplied latitude/longitude.
    """

    conditions = [
        """
        ST_DWithin(
            geog,
            ST_SetSRID(
                ST_MakePoint(:longitude, :latitude),
                4326
            )::geography,
            :radius_m
        )
        """
    ]

    params: dict[str, Any] = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "radius_m": request.radius_km * 1000,
        "limit": request.limit,
    }

    if request.city:
        conditions.append("city ILIKE :city")
        params["city"] = f"%{request.city}%"

    where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            id,
            company_id,
            label,
            area,
            city,
            latitude,
            longitude,
            ROUND(
                (
                    ST_Distance(
                        geog,
                        ST_SetSRID(
                            ST_MakePoint(:longitude, :latitude),
                            4326
                        )::geography
                    ) / 1000.0
                )::numeric,
                3
            ) AS distance_km
        FROM v_company_locations
        {where_clause}
        ORDER BY distance_km ASC
        LIMIT :limit
    """

    return await db.fetch_all(sql, params)