"""End-to-end ingestion against the real migrated database (throwaway test container).

The pipeline commits its own transactions, so each test truncates the data tables first."""

from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.ingestion.loaders import IngestionFileError
from app.ingestion.pipeline import IngestionPipeline

pytestmark = pytest.mark.integration
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "samples" / "companies_sample.csv"

@pytest.fixture
async def pipeline(migrated_db):
    engine = create_async_engine(str(migrated_db["url"]))

    async def reset_data() -> None:
        async with engine.begin() as c:
            await c.execute(
                text(
                    "TRUNCATE entity_match_candidates, coworking_tenants, "
                    "coworking_branches, coworking_buildings, "
                    "coworking_operators, company_locations, "
                    "company_identifiers, company_aliases, companies, "
                    "ingestion_runs, sources CASCADE"
                )
            )

    # Clean before the test.
    await reset_data()

    settings = Settings(
        app_env="test",
        postgres_password=SecretStr("test-pass"),
        ingest_checkpoint_every=2,
    )
    sf = async_sessionmaker(engine, expire_on_commit=False)

    try:
        yield IngestionPipeline(sf, settings), sf
    finally:
        # Clean after the test so committed pipeline data cannot
        # leak into unrelated integration tests.
        await reset_data()
        await engine.dispose()


async def _count(sf, table: str) -> int:
    async with sf() as s:
        return (await s.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()  # noqa: S608


async def test_sample_file_outcomes(pipeline) -> None:
    pipe, sf = pipeline
    report = await pipe.ingest_file(SAMPLE)
    assert report.rows_total == 7
    assert report.failed == 2
    assert {i.kind for i in report.issues} >= {"validation_error", "needs_review"}
    assert report.needs_review == 1
    async with sf() as s:
        n = (await s.execute(text(
            "SELECT count(*) FROM companies WHERE domain='novalabs.example'"))).scalar_one()
    assert n == 1
    assert report.tenancies_upserted == 2


async def test_reingest_is_idempotent(pipeline) -> None:
    pipe, sf = pipeline
    await pipe.ingest_file(SAMPLE)
    companies = await _count(sf, "companies")
    tenants = await _count(sf, "coworking_tenants")
    reviews = await _count(sf, "entity_match_candidates")
    second = await pipe.ingest_file(SAMPLE)
    assert second.inserted == 0
    assert await _count(sf, "companies") == companies
    assert await _count(sf, "coworking_tenants") == tenants
    assert await _count(sf, "entity_match_candidates") == reviews


async def test_postgis_geometry_created_from_coordinates(pipeline) -> None:
    pipe, sf = pipeline
    await pipe.ingest_file(SAMPLE)
    async with sf() as s:
        cnt = (await s.execute(text(
            "SELECT count(*) FROM company_locations WHERE geog IS NOT NULL AND ST_DWithin("
            "geog, ST_SetSRID(ST_MakePoint(77.75,12.9698),4326)::geography, 500)"
        ))).scalar_one()
    assert cnt >= 1


async def test_invalid_phone_dropped_but_row_ingested(pipeline) -> None:
    pipe, sf = pipeline
    await pipe.ingest_file(SAMPLE)
    async with sf() as s:
        phone = (await s.execute(text(
            "SELECT contact_phone FROM coworking_tenants t JOIN companies c ON c.id=t.company_id "
            "WHERE c.domain='zephyr.example'"
        ))).scalar_one()
    assert phone is None


async def test_run_recorded_with_report(pipeline) -> None:
    pipe, sf = pipeline
    await pipe.ingest_file(SAMPLE)
    async with sf() as s:
        status, rep = (await s.execute(text("SELECT status, report FROM ingestion_runs"))).one()
    assert status == "partial"
    assert rep["failed"] == 2


async def test_bad_file_fails_before_touching_db(pipeline, tmp_path: Path) -> None:
    pipe, sf = pipeline
    bad = tmp_path / "bad.csv"
    bad.write_text("Website,City\nx.example,Pune\n")
    with pytest.raises(IngestionFileError):
        await pipe.ingest_file(bad)
    assert await _count(sf, "ingestion_runs") == 0

async def test_coworking_operator_location_resolution(pipeline, tmp_path) -> None:
    pipe, sf = pipeline

    test_file = tmp_path / "coworking_operator_locations.csv"
    test_file.write_text(
        """company_name,address,city,latitude,longitude
WeWork Galaxy,MG Road Bengaluru,Bengaluru,12.9716,77.5946
WeWork Galaxy Meeting Rooms,MG Road Bengaluru,Bengaluru,12.9716,77.5946
WeWork Bellandur,Bellandur Main Road,Bengaluru,12.9298,77.6848
IndiQube Inorbit,inorbit mall rd,Hyderabad,17.4492,78.3815
Redbrick Offices - Cyber Pearl IT Park,inorbit mall rd,Hyderabad,17.4493,78.3816
""",
        encoding="utf-8",
    )

    report = await pipe.ingest_file(test_file)

    assert report.failed == 0

    async with sf() as s:
        # 3 canonical companies:
        # 1 WeWork + 1 IndiQube + 1 Redbrick Offices
        companies = (
            await s.execute(text("SELECT count(*) FROM companies"))
        ).scalar_one()
        assert companies == 3

        # WeWork has two physical locations:
        # one shared/duplicate address and one different address.
        wework_locations = (
            await s.execute(
                text(
                    """
                    SELECT count(*)
                    FROM company_locations cl
                    JOIN companies c ON c.id = cl.company_id
                    WHERE c.normalized_name LIKE 'wework%'
                    """
                )
            )
        ).scalar_one()
        assert wework_locations == 2

        # IndiQube and Redbrick must remain separate companies
        # even though their address strings are identical.
        separate_operator_companies = (
            await s.execute(
                text(
                    """
                    SELECT count(*)
                    FROM companies
                    WHERE normalized_name IN (
                        'indiqube inorbit',
                        'redbrick offices cyber pearl it park'
                    )
                    """
                )
            )
        ).scalar_one()
        assert separate_operator_companies == 2