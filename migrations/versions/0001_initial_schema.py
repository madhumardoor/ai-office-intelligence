"""initial schema

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from alembic import op

import app.models  # noqa: F401
from app.database.base import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Tables that carry updated_at and need the trigger (covers raw-SQL updates, not just ORM).
_TRIGGER_TABLES = [
    "sources", "ingestion_runs", "companies", "company_aliases", "company_identifiers",
    "company_locations", "company_employees", "company_hiring_signals", "company_funding",
    "company_news", "entity_match_candidates", "coworking_operators", "coworking_buildings",
    "coworking_branches", "coworking_tenants", "properties", "documents", "document_chunks",
    "embedding_metadata", "signal_definitions", "signals", "company_signal_scores",
    "search_logs", "llm_logs", "web_search_cache", "users", "api_keys", "conversations", "messages",
]

# Allow-listed, PII-free views. The agent role can read ONLY these.
_VIEWS = {
    "v_companies": """
        SELECT id, name, domain, website, industry, employee_count, city, state
        FROM companies WHERE deleted_at IS NULL""",
    "v_company_locations": """
        SELECT cl.id, cl.company_id, cl.label, cl.area, cl.city, cl.latitude, cl.longitude, cl.geog
        FROM company_locations cl JOIN companies c ON c.id = cl.company_id AND c.deleted_at IS NULL""",
    "v_coworking_branches": """
        SELECT br.id AS branch_id, br.name AS branch_name, br.seat_capacity,
               op.name AS operator_name, b.id AS building_id, b.name AS building_name,
               b.area, b.city, b.latitude, b.longitude, b.geog
        FROM coworking_branches br
        JOIN coworking_operators op ON op.id = br.operator_id
        JOIN coworking_buildings b ON b.id = br.building_id""",
    "v_coworking_tenants": """
        SELECT t.id, t.company_id, c.name AS company_name, t.branch_id, t.seats_used,
               t.first_seen_on, t.last_seen_on
        FROM coworking_tenants t JOIN companies c ON c.id = t.company_id AND c.deleted_at IS NULL""",
    "v_company_employees": """
        SELECT company_id, observed_on, employee_count, location_scope FROM company_employees""",
    "v_company_hiring": """
        SELECT company_id, observed_on, city, open_roles, leadership_roles, evidence_url
        FROM company_hiring_signals""",
    "v_company_funding": """
        SELECT company_id, announced_on, round_name, amount_usd, evidence_url FROM company_funding""",
    "v_company_news": """
        SELECT company_id, title, url, published_at, snippet, category FROM company_news""",
    "v_properties": """
        SELECT id, name, property_type, area, city, latitude, longitude, geog, total_area_sqft
        FROM properties""",
    "v_signals": """
        SELECT s.id, s.company_id, s.signal_code, s.kind, s.evidence_text, s.evidence_url,
               s.detected_on, s.value, s.confidence, src.name AS source_name
        FROM signals s LEFT JOIN sources src ON src.id = s.source_id""",
    "v_signal_scores": """
        SELECT DISTINCT ON (company_id) company_id, score, confidence, signal_count, breakdown,
               config_version, computed_at
        FROM company_signal_scores ORDER BY company_id, computed_at DESC""",
}


def upgrade() -> None:
    for ext in ("postgis", "vector", "pg_trgm", "pgcrypto"):
        op.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)

    # updated_at trigger
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
        BEGIN NEW.updated_at = now(); RETURN NEW; END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in _TRIGGER_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table}_updated_at BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
        )

    # Read-only agent views + grants (role created by docker/postgres/init/02-roles.sh;
    # skipped gracefully where the role does not exist, e.g. bare test databases).
    for name, body in _VIEWS.items():
        op.execute(f"CREATE OR REPLACE VIEW {name} AS {body}")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
                GRANT USAGE ON SCHEMA public TO agent_ro;
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;
            END IF;
        END $$;
        """
    )
    # The blanket grant above would expose base tables, so revoke them and keep views only.
    tables_csv = ", ".join(sorted(t.name for t in Base.metadata.sorted_tables))
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
                REVOKE ALL ON {tables_csv} FROM agent_ro;
                REVOKE ALL ON spatial_ref_sys, geometry_columns, geography_columns FROM agent_ro;
                GRANT SELECT ON spatial_ref_sys, geometry_columns, geography_columns TO agent_ro;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    for name in _VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {name}")
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")