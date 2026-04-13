"""initial schema

Revision ID: fb413cb12cac
Revises:
Create Date: 2026-04-13 21:57:52.192810

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fb413cb12cac"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all tables for the NovelRAG schema."""

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
    )

    # --- workspaces ---
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("language", sa.String(), nullable=False, server_default="zh"),
        sa.Column("beliefs", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )

    # --- resource_aspects ---
    op.create_table(
        "resource_aspects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("uri", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("children_keys", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("root_element_names", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.UniqueConstraint("workspace_id", "uri", name="uq_resource_aspects_workspace_uri"),
    )

    # --- resource_elements (partitioned by workspace_id) ---
    # Note: SQLAlchemy's op.create_table does not support PARTITION BY directly,
    # so we use raw SQL for the partitioned table.
    op.execute("""
        CREATE TABLE resource_elements (
            id SERIAL,
            workspace_id INTEGER NOT NULL,
            aspect_id INTEGER NOT NULL REFERENCES resource_aspects(id),
            name VARCHAR NOT NULL,
            uri VARCHAR NOT NULL,
            relationships JSONB NOT NULL DEFAULT '{}',
            data JSONB NOT NULL DEFAULT '{}',
            embedding vector(3072) NOT NULL,
            PRIMARY KEY (id, workspace_id),
            UNIQUE (workspace_id, aspect_id, uri)
        ) PARTITION BY LIST (workspace_id)
    """)

    # HNSW index on embeddings for cosine similarity search
    op.execute("""
        CREATE INDEX ix_resource_elements_embedding
        ON resource_elements
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # --- backlog_entries ---
    op.create_table(
        "backlog_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
        ),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
    )

    # --- undo_redo_table ---
    op.create_table(
        "undo_redo_table",
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("undo_stack", sa.ARRAY(sa.Integer()), nullable=False, server_default="{}"),
        sa.Column("last_undo_group", sa.String(), nullable=True),
        sa.Column("redo_stack", sa.ARRAY(sa.Integer()), nullable=False, server_default="{}"),
        sa.Column("curr_redo_group", sa.String(), nullable=True),
    )

    # --- undo_items ---
    op.create_table(
        "undo_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
        ),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("group", sa.String(), nullable=True),
    )

    # --- redo_items ---
    op.create_table(
        "redo_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
        ),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("group", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("redo_items")
    op.drop_table("undo_items")
    op.drop_table("undo_redo_table")
    op.drop_table("backlog_entries")
    op.execute("DROP TABLE IF EXISTS resource_elements")
    op.drop_table("resource_aspects")
    op.drop_table("workspaces")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")
