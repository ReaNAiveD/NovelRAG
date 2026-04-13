"""PostgreSQL storage backend for NovelRAG.

Provides workspace-scoped persistence for resources, undo/redo, and backlog
using ``psycopg`` (async) with ``pgvector`` for vector similarity search.

Usage::

    from novelrag.storage.pg import PostgresResourceRepository, PostgresUndoQueue, PostgresBacklog

Database schema is managed via Alembic migrations.  Run from the
``novelrag/storage/pg`` directory::

    NOVELRAG_PG_URL=postgresql+psycopg://... alembic upgrade head
"""

from novelrag.storage.pg.backlog import PostgresBacklog
from novelrag.storage.pg.resource import PostgresResourceRepository
from novelrag.storage.pg.undo import PostgresUndoQueue

__all__ = [
    "PostgresBacklog",
    "PostgresResourceRepository",
    "PostgresUndoQueue",
]
