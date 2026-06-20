"""add model metadata to messages

Revision ID: b6a8d3f92c4d
Revises: 26f682471bc7
Create Date: 2026-06-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6a8d3f92c4d"
down_revision: str | Sequence[str] | None = "26f682471bc7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("messages", sa.Column("provider", sa.String(length=50), nullable=True))
    op.add_column("messages", sa.Column("model", sa.String(length=200), nullable=True))
    op.add_column("messages", sa.Column("usage_json", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("thinking_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("messages", "thinking_json")
    op.drop_column("messages", "usage_json")
    op.drop_column("messages", "model")
    op.drop_column("messages", "provider")
