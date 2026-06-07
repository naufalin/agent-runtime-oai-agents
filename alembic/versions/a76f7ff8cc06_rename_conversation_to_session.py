"""rename conversation to session

Revision ID: a76f7ff8cc06
Revises: 014d52c9da5e
Create Date: 2026-06-07 11:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a76f7ff8cc06"
down_revision: str | Sequence[str] | None = "014d52c9da5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Drop the old FK constraint on messages.conversation_id
    op.drop_constraint("messages_conversation_id_fkey", "messages", type_="foreignkey")

    # 2. Rename conversations table → sessions
    op.rename_table("conversations", "sessions")

    # 3. Rename messages.conversation_id → session_id
    op.alter_column("messages", "conversation_id", new_column_name="session_id")

    # 4. Recreate FK: messages.session_id → sessions.id
    op.create_foreign_key(
        "messages_session_id_fkey",
        "messages",
        "sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 5. Rename the sequence
    op.execute("ALTER SEQUENCE conversations_id_seq RENAME TO sessions_id_seq")


def downgrade() -> None:
    op.drop_constraint("messages_session_id_fkey", "messages", type_="foreignkey")
    op.alter_column("messages", "session_id", new_column_name="conversation_id")
    op.rename_table("sessions", "conversations")
    op.create_foreign_key(
        "messages_conversation_id_fkey",
        "messages",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.execute("ALTER SEQUENCE sessions_id_seq RENAME TO conversations_id_seq")
