import logging
from typing import Any

from psycopg import AsyncCursor
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from novelrag.resource_agent.undo import ReversibleAction, UndoQueue

logger = logging.getLogger(__name__)

# Type alias for the control row dict returned by _ensure_control_row.
_ControlRow = dict[str, Any]


def _row_to_action(row: dict[str, Any]) -> ReversibleAction:
    return ReversibleAction(method=row["method"], params=row["params"], group=row["group"])


def _collect_contiguous_tail(
    stack: list[int],
    lookup: dict[int, dict[str, Any]],
    top_group: str,
) -> tuple[list[int], list[int]]:
    """Walk backward through *stack*, collecting the contiguous tail whose
    group matches *top_group*.  Group names may be duplicated elsewhere in
    the stack; only the uninterrupted run at the end is collected.

    Returns ``(removed_ids, new_stack)``."""
    removed_ids: list[int] = []
    for item_id in reversed(stack):
        row = lookup.get(item_id)
        if row is not None and row["group"] == top_group:
            removed_ids.append(item_id)
        else:
            break
    remove_set = set(removed_ids)
    new_stack = [sid for sid in stack if sid not in remove_set]
    return removed_ids, new_stack


class PostgresUndoQueue(UndoQueue):
    def __init__(self, workspace_id: int, pool: AsyncConnectionPool) -> None:
        self.workspace_id = workspace_id
        self.pool = pool

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _ensure_control_row(self, cur: AsyncCursor[dict[str, Any]]) -> _ControlRow:
        """Return the control row for this workspace, creating it if absent.
        The row is locked with FOR UPDATE for safe concurrent access."""
        await cur.execute(
            """
            INSERT INTO undo_redo_table (workspace_id, undo_stack, last_undo_group, redo_stack, curr_redo_group)
            VALUES (%(ws)s, '{}', NULL, '{}', NULL)
            ON CONFLICT (workspace_id) DO NOTHING
        """,
            {"ws": self.workspace_id},
        )
        await cur.execute(
            """
            SELECT undo_stack, last_undo_group, redo_stack, curr_redo_group
            FROM undo_redo_table WHERE workspace_id = %(ws)s FOR UPDATE
        """,
            {"ws": self.workspace_id},
        )
        row = await cur.fetchone()
        assert row is not None
        return row

    # -- Undo inner methods -------------------------------------------- #

    async def _pop_undo_item_inner(
        self,
        cur: AsyncCursor[dict[str, Any]],
        ctrl: _ControlRow,
    ) -> ReversibleAction | None:
        """Pop and delete the topmost undo item.  Updates the control row."""
        stack: list[int] = ctrl["undo_stack"]
        if not stack:
            return None
        popped_id = stack[-1]
        new_stack = stack[:-1]

        await cur.execute(
            """
            DELETE FROM undo_items WHERE id = %(id)s
            RETURNING method, params, "group"
        """,
            {"id": popped_id},
        )
        result = await cur.fetchone()
        if result is None:
            return None

        new_group: str | None = None
        if new_stack:
            await cur.execute(
                """
                SELECT "group" FROM undo_items WHERE id = %(id)s
            """,
                {"id": new_stack[-1]},
            )
            top = await cur.fetchone()
            if top is not None:
                new_group = top["group"]

        await cur.execute(
            """
            UPDATE undo_redo_table
            SET undo_stack = %(stack)s, last_undo_group = %(grp)s
            WHERE workspace_id = %(ws)s
        """,
            {"stack": new_stack, "grp": new_group, "ws": self.workspace_id},
        )

        return _row_to_action(result)

    async def _pop_undo_group_inner(
        self,
        cur: AsyncCursor[dict[str, Any]],
        ctrl: _ControlRow,
    ) -> list[ReversibleAction] | None:
        """Pop the contiguous tail of undo items sharing the topmost group."""
        stack: list[int] = ctrl["undo_stack"]
        if not stack:
            return None

        top_group: str | None = ctrl["last_undo_group"]
        if top_group is None:
            item = await self._pop_undo_item_inner(cur, ctrl)
            return [item] if item is not None else None

        # Batch-fetch all items in the stack.
        await cur.execute(
            """
            SELECT id, method, params, "group" FROM undo_items
            WHERE id = ANY(%(ids)s)
        """,
            {"ids": stack},
        )
        rows = await cur.fetchall()
        lookup: dict[int, dict[str, Any]] = {r["id"]: r for r in rows}

        removed_ids, new_stack = _collect_contiguous_tail(stack, lookup, top_group)
        if not removed_ids:
            return None

        result = [_row_to_action(lookup[rid]) for rid in removed_ids]

        await cur.execute(
            """
            DELETE FROM undo_items WHERE id = ANY(%(ids)s)
        """,
            {"ids": removed_ids},
        )

        # Determine new top-of-stack group.
        new_group: str | None = None
        if new_stack:
            top_row = lookup.get(new_stack[-1])
            if top_row is not None:
                new_group = top_row["group"]
            else:
                await cur.execute(
                    """
                    SELECT "group" FROM undo_items WHERE id = %(id)s
                """,
                    {"id": new_stack[-1]},
                )
                fetched = await cur.fetchone()
                if fetched is not None:
                    new_group = fetched["group"]

        await cur.execute(
            """
            UPDATE undo_redo_table
            SET undo_stack = %(stack)s, last_undo_group = %(grp)s
            WHERE workspace_id = %(ws)s
        """,
            {"stack": new_stack, "grp": new_group, "ws": self.workspace_id},
        )

        return result

    # -- Redo inner methods -------------------------------------------- #

    async def _pop_redo_item_inner(
        self,
        cur: AsyncCursor[dict[str, Any]],
        ctrl: _ControlRow,
    ) -> ReversibleAction | None:
        """Pop and delete the topmost redo item.  Updates the control row."""
        stack: list[int] = ctrl["redo_stack"]
        if not stack:
            return None
        popped_id = stack[-1]
        new_stack = stack[:-1]

        await cur.execute(
            """
            DELETE FROM redo_items WHERE id = %(id)s
            RETURNING method, params, "group"
        """,
            {"id": popped_id},
        )
        result = await cur.fetchone()
        if result is None:
            return None

        new_group: str | None = None
        if new_stack:
            await cur.execute(
                """
                SELECT "group" FROM redo_items WHERE id = %(id)s
            """,
                {"id": new_stack[-1]},
            )
            top = await cur.fetchone()
            if top is not None:
                new_group = top["group"]

        await cur.execute(
            """
            UPDATE undo_redo_table
            SET redo_stack = %(stack)s, curr_redo_group = %(grp)s
            WHERE workspace_id = %(ws)s
        """,
            {"stack": new_stack, "grp": new_group, "ws": self.workspace_id},
        )

        return _row_to_action(result)

    async def _pop_redo_group_inner(
        self,
        cur: AsyncCursor[dict[str, Any]],
        ctrl: _ControlRow,
    ) -> list[ReversibleAction] | None:
        """Pop the contiguous tail of redo items sharing the topmost group."""
        stack: list[int] = ctrl["redo_stack"]
        if not stack:
            return None

        top_group: str | None = ctrl["curr_redo_group"]
        if top_group is None:
            item = await self._pop_redo_item_inner(cur, ctrl)
            return [item] if item is not None else None

        # Batch-fetch all items in the stack.
        await cur.execute(
            """
            SELECT id, method, params, "group" FROM redo_items
            WHERE id = ANY(%(ids)s)
        """,
            {"ids": stack},
        )
        rows = await cur.fetchall()
        lookup: dict[int, dict[str, Any]] = {r["id"]: r for r in rows}

        removed_ids, new_stack = _collect_contiguous_tail(stack, lookup, top_group)
        if not removed_ids:
            return None

        result = [_row_to_action(lookup[rid]) for rid in removed_ids]

        await cur.execute(
            """
            DELETE FROM redo_items WHERE id = ANY(%(ids)s)
        """,
            {"ids": removed_ids},
        )

        # Determine new top-of-stack group.
        new_group: str | None = None
        if new_stack:
            top_row = lookup.get(new_stack[-1])
            if top_row is not None:
                new_group = top_row["group"]
            else:
                await cur.execute(
                    """
                    SELECT "group" FROM redo_items WHERE id = %(id)s
                """,
                    {"id": new_stack[-1]},
                )
                fetched = await cur.fetchone()
                if fetched is not None:
                    new_group = fetched["group"]

        await cur.execute(
            """
            UPDATE undo_redo_table
            SET redo_stack = %(stack)s, curr_redo_group = %(grp)s
            WHERE workspace_id = %(ws)s
        """,
            {"stack": new_stack, "grp": new_group, "ws": self.workspace_id},
        )

        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def add_undo_item(self, item: ReversibleAction, clear_redo: bool = True) -> list[ReversibleAction] | None:
        """
        Add an undo item to the queue.
        Args:
            item: The UndoItem to add.
            clear_redo: Whether to clear the redo stack.
        Returns:
            The overwritten list of RedoItems, if any.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)

            # Insert the new undo item.
            await cur.execute(
                """
                        INSERT INTO undo_items (workspace_id, method, params, "group")
                        VALUES (%(ws)s, %(method)s, %(params)s, %(group)s)
                        RETURNING id
                    """,
                {"ws": self.workspace_id, "method": item.method, "params": item.params, "group": item.group},
            )
            new_row = await cur.fetchone()
            assert new_row is not None
            new_id: int = new_row["id"]

            # Optionally clear the redo stack.
            deleted_items: list[ReversibleAction] | None = None
            redo_stack = ctrl["redo_stack"]
            if clear_redo:
                deleted_items = []
                if redo_stack:
                    await cur.execute(
                        """
                                DELETE FROM redo_items WHERE id = ANY(%(ids)s)
                                RETURNING method, params, "group"
                            """,
                        {"ids": redo_stack},
                    )
                    deleted = await cur.fetchall()
                    deleted_items = [_row_to_action(r) for r in deleted]

            new_undo_stack = ctrl["undo_stack"] + [new_id]
            if clear_redo:
                await cur.execute(
                    """
                            UPDATE undo_redo_table
                            SET undo_stack = %(us)s, last_undo_group = %(ug)s,
                                redo_stack = '{}', curr_redo_group = NULL
                            WHERE workspace_id = %(ws)s
                        """,
                    {"us": new_undo_stack, "ug": item.group, "ws": self.workspace_id},
                )
            else:
                await cur.execute(
                    """
                            UPDATE undo_redo_table
                            SET undo_stack = %(us)s, last_undo_group = %(ug)s
                            WHERE workspace_id = %(ws)s
                        """,
                    {"us": new_undo_stack, "ug": item.group, "ws": self.workspace_id},
                )

            logger.debug(
                "Added undo item(id: %d) to workspace %d: %s (group=%s)",
                new_id,
                self.workspace_id,
                item.method,
                item.group,
            )
            return deleted_items

    async def add_redo_item(self, item: ReversibleAction) -> None:
        """
        Add a redo item to the queue.
        Args:
            item: The RedoItem to add.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)

            await cur.execute(
                """
                        INSERT INTO redo_items (workspace_id, method, params, "group")
                        VALUES (%(ws)s, %(method)s, %(params)s, %(group)s)
                        RETURNING id
                    """,
                {"ws": self.workspace_id, "method": item.method, "params": item.params, "group": item.group},
            )
            new_row = await cur.fetchone()
            assert new_row is not None
            new_id: int = new_row["id"]

            new_redo_stack = ctrl["redo_stack"] + [new_id]
            await cur.execute(
                """
                        UPDATE undo_redo_table
                        SET redo_stack = %(rs)s, curr_redo_group = %(rg)s
                        WHERE workspace_id = %(ws)s
                    """,
                {"rs": new_redo_stack, "rg": item.group, "ws": self.workspace_id},
            )

            logger.debug(
                "Added redo item(id: %d) to workspace %d: %s (group=%s)",
                new_id,
                self.workspace_id,
                item.method,
                item.group,
            )

    async def pop_undo_item(self) -> ReversibleAction | None:
        """
        Pop the last undo item from the queue.
        Returns:
            The last UndoItem, or None if the queue is empty.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)
            return await self._pop_undo_item_inner(cur, ctrl)

    async def pop_undo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of undo items from the queue.
        Returns:
            The list of UndoItems in execution order (newest to oldest —
            iterate forward to undo correctly). Returns None if empty.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)
            return await self._pop_undo_group_inner(cur, ctrl)

    async def pop_redo_item(self) -> ReversibleAction | None:
        """
        Pop the last redo item from the queue.
        Returns:
            The last RedoItem, or None if the queue is empty.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)
            return await self._pop_redo_item_inner(cur, ctrl)

    async def pop_redo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of redo items from the queue.
        The group is ordered from first to last.
        Returns:
            The list of RedoItems in execution order in the last group, or None if the queue is empty.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)
            return await self._pop_redo_group_inner(cur, ctrl)

    async def peek_recent(self, n: int = 5) -> list[ReversibleAction]:
        """
        Peek at the most recent undo items without removing them.
        Args:
            n: Number of recent items to return.
        Returns:
            List of the most recent ReversibleActions (newest first),
            up to n items.
        """
        if n <= 0:
            return []
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                    SELECT undo_stack FROM undo_redo_table
                    WHERE workspace_id = %(ws)s
                """,
                {"ws": self.workspace_id},
            )
            row = await cur.fetchone()
            if row is None or not row["undo_stack"]:
                return []

            tail_ids = row["undo_stack"][-n:]
            await cur.execute(
                """
                    SELECT id, method, params, "group" FROM undo_items
                    WHERE id = ANY(%(ids)s)
                """,
                {"ids": tail_ids},
            )
            rows = await cur.fetchall()
            lookup = {r["id"]: r for r in rows}
            # Return in newest-first order (reverse of stack tail).
            return [_row_to_action(lookup[sid]) for sid in reversed(tail_ids) if sid in lookup]

    async def clear(self) -> None:
        """
        Clear the undo and redo queues.
        """
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            ctrl = await self._ensure_control_row(cur)
            if ctrl["undo_stack"]:
                await cur.execute(
                    """
                            DELETE FROM undo_items WHERE id = ANY(%(ids)s)
                        """,
                    {"ids": ctrl["undo_stack"]},
                )
            if ctrl["redo_stack"]:
                await cur.execute(
                    """
                            DELETE FROM redo_items WHERE id = ANY(%(ids)s)
                        """,
                    {"ids": ctrl["redo_stack"]},
                )
            await cur.execute(
                """
                        UPDATE undo_redo_table
                        SET undo_stack = '{}', last_undo_group = NULL,
                            redo_stack = '{}', curr_redo_group = NULL
                        WHERE workspace_id = %(ws)s
                    """,
                {"ws": self.workspace_id},
            )
