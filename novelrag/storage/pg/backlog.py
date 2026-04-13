import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from novelrag.resource_agent.backlog import Backlog, BacklogEntry

logger = logging.getLogger(__name__)


def _row_to_entry(row: dict[str, Any]) -> BacklogEntry:
    """Map a database row dict to a ``BacklogEntry``."""
    return BacklogEntry(
        type=row["type"],
        priority=row["priority"],
        description=row["description"],
        metadata=row.get("metadata") or {},
    )


class PostgresBacklog(Backlog[BacklogEntry]):
    """Priority-sorted backlog backed by PostgreSQL.

    All queries are scoped to *workspace_id*.  Entries are stored in
    descending priority order via ``ORDER BY priority DESC``.
    """

    def __init__(self, workspace_id: int, pool: AsyncConnectionPool) -> None:
        self.workspace_id = workspace_id
        self.pool = pool
        self._count: int | None = None

    async def add_entry(self, entry: BacklogEntry) -> None:
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO backlog_entries (workspace_id, type, priority, description, metadata) "
                "VALUES (%(ws)s, %(type)s, %(priority)s, %(description)s, %(metadata)s)",
                {
                    "ws": self.workspace_id,
                    "type": entry.type,
                    "priority": entry.priority,
                    "description": entry.description,
                    "metadata": entry.metadata,
                },
            )
        self._count = None  # invalidate cache

    async def get_entries(self) -> list[BacklogEntry]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT type, priority, description, metadata "
                "FROM backlog_entries "
                "WHERE workspace_id = %(ws)s "
                "ORDER BY priority DESC, id ASC",
                {"ws": self.workspace_id},
            )
            rows = await cur.fetchall()
            return [_row_to_entry(r) for r in rows]

    async def clear(self) -> None:
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM backlog_entries WHERE workspace_id = %(ws)s",
                {"ws": self.workspace_id},
            )
        self._count = None

    async def get_top(self, n: int) -> list[BacklogEntry]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT type, priority, description, metadata "
                "FROM backlog_entries "
                "WHERE workspace_id = %(ws)s "
                "ORDER BY priority DESC, id ASC "
                "LIMIT %(n)s",
                {"ws": self.workspace_id, "n": n},
            )
            rows = await cur.fetchall()
            return [_row_to_entry(r) for r in rows]

    async def pop_entry(self) -> BacklogEntry | None:
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "DELETE FROM backlog_entries "
                "WHERE id = ("
                "  SELECT id FROM backlog_entries "
                "  WHERE workspace_id = %(ws)s "
                "  ORDER BY priority DESC, id ASC "
                "  LIMIT 1 "
                "  FOR UPDATE SKIP LOCKED"
                ") RETURNING type, priority, description, metadata",
                {"ws": self.workspace_id},
            )
            row = await cur.fetchone()
            if row is None:
                return None
            self._count = None
            return _row_to_entry(row)

    async def remove_entries(self, indices: list[int]) -> list[BacklogEntry]:
        """Remove entries at the given 0-based indices and return them.

        Indices refer to the current priority-sorted order.
        Out-of-range indices are silently ignored.
        """
        if not indices:
            return []
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            # Fetch all entries in order to map indices to database ids.
            await cur.execute(
                "SELECT id, type, priority, description, metadata "
                "FROM backlog_entries "
                "WHERE workspace_id = %(ws)s "
                "ORDER BY priority DESC, id ASC",
                {"ws": self.workspace_id},
            )
            all_rows = await cur.fetchall()
            valid_indices = sorted(set(idx for idx in indices if 0 <= idx < len(all_rows)))
            if not valid_indices:
                return []
            ids_to_delete = [all_rows[idx]["id"] for idx in valid_indices]
            await cur.execute(
                "DELETE FROM backlog_entries WHERE id = ANY(%(ids)s)",
                {"ids": ids_to_delete},
            )
            self._count = None
            return [_row_to_entry(all_rows[idx]) for idx in valid_indices]

    def __len__(self) -> int:
        # Backlog Protocol requires __len__; use cached count or 0 as fallback.
        # For accurate count, call get_entries() or use _refresh_count().
        return self._count if self._count is not None else 0

    async def refresh_count(self) -> int:
        """Fetch the current entry count from the database."""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM backlog_entries WHERE workspace_id = %(ws)s",
                {"ws": self.workspace_id},
            )
            row = await cur.fetchone()
            self._count = row[0] if row else 0
            return self._count
