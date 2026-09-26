import pytest
from sqlalchemy import text

from app.database.spatial import companies_near_building, companies_near_point, nearest_coworking_buildings

pytestmark = pytest.mark.integration

# Real Bangalore coordinates (approx): Whitefield vs Koramangala are ~17 km apart.
WHITEFIELD = (12.9698, 77.7500)
KORAMANGALA = (12.9352, 77.6245)


async def _company_at(session, name: str, lat: float, lng: float) -> None:
    cid = (await session.execute(text(
        "INSERT INTO companies (name, normalized_name, domain) VALUES (:n, :nn, :d) RETURNING id"),
        {"n": name, "nn": name.lower(), "d": f"{name.lower()}.example"})).scalar_one()
    await session.execute(text(
        "INSERT INTO company_locations (company_id, latitude, longitude, geog, area) "
        "VALUES (:c, :lat, :lng, ST_SetSRID(ST_MakePoint(:lng,:lat),4326)::geography, 'x')"),
        {"c": cid, "lat": lat, "lng": lng})


async def test_radius_search_returns_only_nearby(db_session) -> None:
    await _company_at(db_session, "NearWF", WHITEFIELD[0] + 0.005, WHITEFIELD[1])
    await _company_at(db_session, "FarKor", *KORAMANGALA)
    result = await companies_near_point(db_session, lat=WHITEFIELD[0], lng=WHITEFIELD[1], radius_km=5)
    names = [r.name for r in result]
    assert "NearWF" in names and "FarKor" not in names


async def test_distance_is_in_meters_and_sorted(db_session) -> None:
    await _company_at(db_session, "Close", WHITEFIELD[0] + 0.001, WHITEFIELD[1])
    await _company_at(db_session, "Mid", WHITEFIELD[0] + 0.02, WHITEFIELD[1])
    result = await companies_near_point(db_session, lat=WHITEFIELD[0], lng=WHITEFIELD[1], radius_km=10)
    assert [r.name for r in result] == ["Close", "Mid"]
    assert 90 < result[0].distance_m < 130          # ~111 m per 0.001 deg latitude
    assert 2000 < result[1].distance_m < 2400       # ~2.2 km


async def test_known_distance_whitefield_to_koramangala(db_session) -> None:
    d = (await db_session.execute(text(
        "SELECT ST_Distance(ST_SetSRID(ST_MakePoint(:a,:b),4326)::geography, "
        "ST_SetSRID(ST_MakePoint(:c,:d),4326)::geography)"),
        {"a": WHITEFIELD[1], "b": WHITEFIELD[0], "c": KORAMANGALA[1], "d": KORAMANGALA[0]})).scalar_one()
    assert 13_500 < d < 15_000


async def test_companies_near_building_and_knn(db_session) -> None:
    bid = (await db_session.execute(text(
        "INSERT INTO coworking_buildings (name, city, latitude, longitude, geog) "
        "VALUES ('B1','Bengaluru',:lat,:lng, ST_SetSRID(ST_MakePoint(:lng,:lat),4326)::geography) RETURNING id"),
        {"lat": WHITEFIELD[0], "lng": WHITEFIELD[1]})).scalar_one()
    await db_session.execute(text(
        "INSERT INTO coworking_buildings (name, city, latitude, longitude, geog) "
        "VALUES ('B2','Bengaluru',:lat,:lng, ST_SetSRID(ST_MakePoint(:lng,:lat),4326)::geography)"),
        {"lat": KORAMANGALA[0], "lng": KORAMANGALA[1]})
    await _company_at(db_session, "Adjacent", WHITEFIELD[0] + 0.002, WHITEFIELD[1])
    near = await companies_near_building(db_session, building_id=bid, radius_km=1)
    assert [r.name for r in near] == ["Adjacent"]
    knn = await nearest_coworking_buildings(db_session, lat=WHITEFIELD[0], lng=WHITEFIELD[1], limit=2)
    assert [k["name"] for k in knn] == ["B1", "B2"]


async def test_spatial_index_is_used(db_session) -> None:
    # Force the planner to prefer the index on the tiny test table.
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))
    plan = (await db_session.execute(text(
        "EXPLAIN SELECT 1 FROM company_locations WHERE ST_DWithin(geog, "
        "ST_SetSRID(ST_MakePoint(77.75,12.97),4326)::geography, 5000)"))).all()
    assert any("ix_company_locations_geog" in row[0] for row in plan)


async def test_invalid_radius_rejected(db_session) -> None:
    with pytest.raises(ValueError):
        await companies_near_point(db_session, lat=12.9, lng=77.6, radius_km=0)