"""signal_definitions.half_life_days server default

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE signal_definitions ALTER COLUMN half_life_days SET DEFAULT 90")


def downgrade() -> None:
    op.execute("ALTER TABLE signal_definitions ALTER COLUMN half_life_days DROP DEFAULT")