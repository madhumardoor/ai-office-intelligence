"""geocode cache

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    # On a fresh database, migration 0001 (create_all from current models) already created
    # this table. On an older database it does not exist yet. Handle both.
    if not sa.inspect(bind).has_table("geocode_cache"):
        op.create_table(
            "geocode_cache",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
            sa.Column("query", sa.Text(), nullable=False),
            sa.Column("query_hash", sa.String(64), nullable=False),
            sa.Column("found", sa.Boolean(), nullable=False),
            sa.Column("latitude", sa.Float()),
            sa.Column("longitude", sa.Float()),
            sa.Column("provider", sa.String(30), nullable=False, server_default="nominatim"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("query_hash", name="uq_geocode_cache_query_hash"),
        )
    op.execute("DROP TRIGGER IF EXISTS trg_geocode_cache_updated_at ON geocode_cache")
    op.execute(
        "CREATE TRIGGER trg_geocode_cache_updated_at BEFORE UPDATE ON geocode_cache "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS geocode_cache")