"""add tools_json column to sessions

Revision ID: fd08fd099b38
Revises: faa04ba69d75
Create Date: 2026-06-27 18:55:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fd08fd099b38"
down_revision: str | Sequence[str] | None = "faa04ba69d75"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Per-session tool allowlist:
    #   NULL  -> use server defaults (preserves prior behavior)
    #   []    -> agent has no tools (pure chat)
    #   ["x"] -> named tools only
    op.add_column(
        "sessions",
        sa.Column("tools_json", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sessions", "tools_json")
