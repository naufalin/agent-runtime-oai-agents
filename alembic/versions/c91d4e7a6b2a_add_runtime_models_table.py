"""add runtime models table

Revision ID: c91d4e7a6b2a
Revises: b6a8d3f92c4d
Create Date: 2026-06-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c91d4e7a6b2a"
down_revision: str | Sequence[str] | None = "b6a8d3f92c4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "runtime_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("supports_reasoning", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "model_id",
            name="uq_runtime_models_provider_model_id",
        ),
    )

    runtime_models = sa.table(
        "runtime_models",
        sa.column("provider", sa.String),
        sa.column("model_id", sa.String),
        sa.column("name", sa.String),
        sa.column("enabled", sa.Boolean),
        sa.column("supports_reasoning", sa.Boolean),
        sa.column("sort_order", sa.Integer),
    )
    op.bulk_insert(
        runtime_models,
        [
            {
                "provider": "openrouter",
                "model_id": "z-ai/glm-5.2",
                "name": "Z.ai: GLM 5.2",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 10,
            },
            {
                "provider": "openrouter",
                "model_id": "qwen/qwen3.7-max",
                "name": "Qwen: Qwen3.7 Max",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 20,
            },
            {
                "provider": "openrouter",
                "model_id": "qwen/qwen3.7-plus",
                "name": "Qwen: Qwen3.7 Plus",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 30,
            },
            {
                "provider": "openrouter",
                "model_id": "moonshotai/kimi-k2.7-code",
                "name": "MoonshotAI: Kimi K2.7 Code",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 40,
            },
            {
                "provider": "openrouter",
                "model_id": "minimax/minimax-m3",
                "name": "MiniMax: MiniMax M3",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 50,
            },
            {
                "provider": "openrouter",
                "model_id": "deepseek/deepseek-v4-pro",
                "name": "DeepSeek: DeepSeek V4 Pro",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 60,
            },
            {
                "provider": "openrouter",
                "model_id": "deepseek/deepseek-v4-flash",
                "name": "DeepSeek: DeepSeek V4 Flash",
                "enabled": True,
                "supports_reasoning": True,
                "sort_order": 70,
            },
        ],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("runtime_models")
