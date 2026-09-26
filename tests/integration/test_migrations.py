import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "companies", "company_locations", "company_employees", "company_hiring_signals", "company_funding",
    "company_news", "coworking_operators", "coworking_branches", "coworking_buildings", "coworking_tenants",
    "properties", "documents", "document_chunks", "embedding_metadata", "signals", "company_signal_scores",
    "sources", "search_logs", "llm_logs", "users", "conversations", "messages", "signal_definitions",
    "ingestion_runs", "entity_match_candidates", "company_aliases", "company_identifiers",
    "web_search_cache", "api_keys",
}


async def test_all_tables_exist(db_session) -> None:
    rows = await db_session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
    assert EXPECTED_TABLES <= {r[0] for r in rows}


async def test_critical_indexes_exist(db_session) -> None:
    rows = await db_session.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public'"))
    defs = {r[0]: r[1] for r in rows}
    assert "USING gist" in defs["ix_company_locations_geog"]
    assert "USING hnsw" in defs["ix_document_chunks_embedding_hnsw"]
    assert "vector_cosine_ops" in defs["ix_document_chunks_embedding_hnsw"]
    assert "USING gin" in defs["ix_document_chunks_tsv"]
    assert "gin_trgm_ops" in defs["ix_companies_name_trgm"]


async def test_updated_at_trigger_fires(db_session) -> None:
    await db_session.execute(text("INSERT INTO sources (source_type, name) VALUES ('manual','trg-test')"))
    before = (await db_session.execute(text("SELECT updated_at FROM sources WHERE name='trg-test'"))).scalar_one()
    await db_session.execute(text("SELECT pg_sleep(0.05)"))
    await db_session.execute(text("UPDATE sources SET reliability = 0.9 WHERE name='trg-test'"))
    after = (await db_session.execute(text("SELECT updated_at FROM sources WHERE name='trg-test'"))).scalar_one()
    assert after >= before