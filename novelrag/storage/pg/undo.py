import logging

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue

logger = logging.getLogger(__name__)


class PostgresUndoQueue(UndoQueue):
    def __init__(self, workspace_id: int, pool: AsyncConnectionPool) -> None:
        self.workspace_id = workspace_id
        self.pool = pool

    async def add_undo_item(self, item: ReversibleAction, clear_redo: bool = True) -> list[ReversibleAction] | None:
        """
        Add an undo item to the queue.
        Args:
            item: The UndoItem to add.
            clear_redo: Whether to clear the redo stack.
        Returns:
            The overwritten list of RedoItems, if any.
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                deleted_items = None
                if clear_redo:
                    async with conn.cursor(row_factory=dict_row) as cur:
                        await cur.execute('''
                            DELETE FROM redo_items WHERE workspace_id = %(ws)s
                            RETURNING method, params, "group"
                        ''', {'ws': self.workspace_id})
                        deleted = await cur.fetchall()
                        deleted_items = [ReversibleAction(method=r['method'], params=r['params'], group=r['group']) for r in deleted]
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute('''
                        INSERT INTO undo_items (workspace_id, method, params, "group")
                        VALUES (%(ws)s, %(method)s, %(params)s, %(group)s)
                        RETURNING id, method, params, "group"
                    ''', {'ws': self.workspace_id, 'method': item.method, 'params': item.params, 'group': item.group})
                    result = await cur.fetchone()
                    assert result is not None  # Should never be None since we're inserting
                    logger.debug(f"Added undo item(id: {result['id']}) to workspace {self.workspace_id}: {result['method']} with params {result['params']} in group {result['group']}")
                return deleted_items

    async def add_redo_item(self, item: ReversibleAction) -> None:
        """
        Add a redo item to the queue.
        Args:
            item: The RedoItem to add.
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute('''
                        INSERT INTO redo_items (workspace_id, method, params, "group")
                        VALUES (%(ws)s, %(method)s, %(params)s, %(group)s)
                        RETURNING id, method, params, "group"
                    ''', {'ws': self.workspace_id, 'method': item.method, 'params': item.params, 'group': item.group})
                    result = await cur.fetchone()
                    assert result is not None  # Should never be None since we're inserting
                    logger.debug(f"Added redo item(id: {result['id']}) to workspace {self.workspace_id}: {result['method']} with params {result['params']} in group {result['group']}")

    async def pop_undo_item(self) -> ReversibleAction | None:
        """
        Pop the last undo item from the queue.
        Returns:
            The last UndoItem, or None if the queue is empty.
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute('''
                        DELETE FROM undo_items
                        WHERE id = (
                            SELECT id FROM undo_items
                            WHERE workspace_id = %(ws)s ORDER BY id DESC LIMIT 1
                        )
                        RETURNING method, params, "group"
                    ''', {'ws': self.workspace_id})
                    result = await cur.fetchone()
                    if result is None:
                        return None
                    return ReversibleAction(method=result['method'], params=result['params'], group=result['group'])

    async def pop_undo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of undo items from the queue.
        Returns:
            The list of UndoItems in execution order (newest to oldest — 
            iterate forward to undo correctly). Returns None if empty.
        """
        pass

    async def pop_redo_item(self) -> ReversibleAction | None:
        """
        Pop the last redo item from the queue.
        Returns:
            The last RedoItem, or None if the queue is empty.
        """
        pass

    async def pop_redo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of redo items from the queue.
        The group is ordered from first to last.
        Returns:
            The list of RedoItems in execution order in the last group, or None if the queue is empty.
        """
        pass

    async def peek_recent(self, n: int = 5) -> list[ReversibleAction]:
        """
        Peek at the most recent undo items without removing them.
        Args:
            n: Number of recent items to return.
        Returns:
            List of the most recent ReversibleActions (newest first),
            up to n items.
        """
        pass

    async def clear(self) -> None:
        """
        Clear the undo and redo queues.
        """
        pass
