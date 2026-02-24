from asyncpg import Pool
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue


class PostgresUndoQueue(UndoQueue):
    def __init__(self, pool: Pool) -> None:
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
        pass

    async def add_redo_item(self, item: ReversibleAction) -> None:
        """
        Add a redo item to the queue.
        Args:
            item: The RedoItem to add.
        """
        pass

    async def pop_undo_item(self) -> ReversibleAction | None:
        """
        Pop the last undo item from the queue.
        Returns:
            The last UndoItem, or None if the queue is empty.
        """
        pass
    
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
