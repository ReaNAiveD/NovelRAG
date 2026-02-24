from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ReversibleAction:
    method: str
    params: dict[str, Any]
    group: str | None = None

    @property
    def description(self) -> str:
        """Create a human-readable description of this reversible action."""
        match self.method:
            case 'apply':
                op = self.params.get('op', {})
                target = op.get('target', 'unknown')
                if target == 'property':
                    uri = op.get('resource_uri', 'unknown')
                    keys = list(op.get('data', {}).keys())
                    if keys:
                        return f"Property update on {uri} (fields: {', '.join(keys)})"
                    return f"Property update on {uri}"
                elif target == 'resource':
                    loc = op.get('location', {})
                    uri = loc.get('resource_uri', 'unknown')
                    children_key = loc.get('children_key')
                    loc_label = f"{uri}/{children_key}" if children_key else uri
                    data = op.get('data')
                    start, end = op.get('start', 0), op.get('end', 0)
                    if data and end == start:
                        return f"Inserted {len(data)} resource(s) at {loc_label}"
                    elif not data and end > start:
                        return f"Removed {end - start} resource(s) from {loc_label}"
                    else:
                        return f"Spliced resources at {loc_label}"
                return "Applied operation on repository"
            case 'update_relationships':
                src = self.params.get('source_uri', 'unknown')
                tgt = self.params.get('target_uri', 'unknown')
                rels = self.params.get('relationships', [])
                return f"Updated relationships between {src} and {tgt} ({len(rels)} relation(s))"
            case 'add_aspect':
                name = self.params.get('name', 'unknown')
                return f"Added aspect '{name}'"
            case 'remove_aspect':
                name = self.params.get('name', 'unknown')
                return f"Removed aspect '{name}'"
            case _:
                return f"Unknown action: {self.method}"


class UndoQueue(Protocol):
    async def add_undo_item(self, item: ReversibleAction, clear_redo: bool = True) -> list[ReversibleAction] | None:
        """
        Add an undo item to the queue.
        Args:
            item: The UndoItem to add.
            clear_redo: Whether to clear the redo stack.
        Returns:
            The overwritten list of RedoItems, if any.
        """
        ...

    async def add_redo_item(self, item: ReversibleAction) -> None:
        """
        Add a redo item to the queue.
        Args:
            item: The RedoItem to add.
        """
        ...

    async def pop_undo_item(self) -> ReversibleAction | None:
        """
        Pop the last undo item from the queue.
        Returns:
            The last UndoItem, or None if the queue is empty.
        """
        ...
    
    async def pop_undo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of undo items from the queue.
        Returns:
            The list of UndoItems in execution order (newest to oldest — 
            iterate forward to undo correctly). Returns None if empty.
        """
        ...

    async def pop_redo_item(self) -> ReversibleAction | None:
        """
        Pop the last redo item from the queue.
        Returns:
            The last RedoItem, or None if the queue is empty.
        """
        ...

    async def pop_redo_group(self) -> list[ReversibleAction] | None:
        """
        Pop the last group of redo items from the queue.
        The group is ordered from first to last.
        Returns:
            The list of RedoItems in execution order in the last group, or None if the queue is empty.
        """
        ...

    async def peek_recent(self, n: int = 5) -> list[ReversibleAction]:
        """
        Peek at the most recent undo items without removing them.
        Args:
            n: Number of recent items to return.
        Returns:
            List of the most recent ReversibleActions (newest first),
            up to n items.
        """
        ...

    async def clear(self) -> None:
        """
        Clear the undo and redo queues.
        """
        ...


class MemoryUndoQueue(UndoQueue):
    def __init__(self, undo_stack: list[ReversibleAction] | None = None,
                 redo_stack: list[ReversibleAction] | None = None, stack_size: int | None = 100) -> None:
        self.undo_stack: list[ReversibleAction] = undo_stack if undo_stack is not None else []
        self.redo_stack: list[ReversibleAction] = redo_stack if redo_stack is not None else []
        self.stack_size = stack_size
    
    async def add_undo_item(self, item: ReversibleAction, clear_redo: bool = True) -> list[ReversibleAction] | None:
        self.undo_stack.append(item)
        if self.stack_size is not None and len(self.undo_stack) > self.stack_size:
            self.undo_stack.pop(0)
        overwritten_redo = None
        if clear_redo:
            overwritten_redo = self.redo_stack.copy()
            self.redo_stack = []
        return overwritten_redo

    async def add_redo_item(self, item: ReversibleAction) -> None:
        self.redo_stack.append(item)
        if self.stack_size is not None and len(self.redo_stack) > self.stack_size:
            self.redo_stack.pop(0)

    async def pop_undo_item(self) -> ReversibleAction | None:
        if not self.undo_stack:
            return None
        return self.undo_stack.pop()

    async def pop_undo_group(self) -> list[ReversibleAction] | None:
        if not self.undo_stack:
            return None
        group = []
        last_group = self.undo_stack[-1].group
        if last_group is None:
            return [self.undo_stack.pop()]
        while self.undo_stack and self.undo_stack[-1].group == last_group:
            group.append(self.undo_stack.pop())
        return group

    async def pop_redo_item(self) -> ReversibleAction | None:
        if not self.redo_stack:
            return None
        return self.redo_stack.pop()

    async def pop_redo_group(self) -> list[ReversibleAction] | None:
        if not self.redo_stack:
            return None
        group = []
        last_group = self.redo_stack[-1].group
        if last_group is None:
            return [self.redo_stack.pop()]
        while self.redo_stack and self.redo_stack[-1].group == last_group:
            group.append(self.redo_stack.pop())
        return group

    async def peek_recent(self, n: int = 5) -> list[ReversibleAction]:
        if n <= 0:
            return []
        if not self.undo_stack:
            return []
        return list(reversed(self.undo_stack[-n:]))

    async def clear(self) -> None:
        self.undo_stack = []
        self.redo_stack = []


class LocalUndoQueue(MemoryUndoQueue):
    def __init__(self, path: str, undo_stack: list[ReversibleAction] | None = None,
                 redo_stack: list[ReversibleAction] | None = None, stack_size: int | None = 100) -> None:
        self.path = path
        super().__init__(undo_stack, redo_stack, stack_size)

    @classmethod
    def load(cls, path: str, stack_size: int | None = 100) -> 'LocalUndoQueue':
        import os
        import json

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            undo_stack = [ReversibleAction(**item) for item in data.get("undo_stack", [])]
            redo_stack = [ReversibleAction(**item) for item in data.get("redo_stack", [])]
            return cls(path, undo_stack, redo_stack, stack_size)
        else:
            return cls(path, stack_size=stack_size)

    def _save(self) -> None:
        import os
        import json

        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            data = {
                "undo_stack": [item.__dict__ for item in self.undo_stack],
                "redo_stack": [item.__dict__ for item in self.redo_stack],
            }
            json.dump(data, f, indent=4, ensure_ascii=False)

    async def add_undo_item(self, item: ReversibleAction, clear_redo: bool = True) -> list[ReversibleAction] | None:
        overwritten_redo = await super().add_undo_item(item, clear_redo)
        self._save()
        return overwritten_redo

    async def add_redo_item(self, item: ReversibleAction) -> None:
        await super().add_redo_item(item)
        self._save()

    async def pop_undo_item(self) -> ReversibleAction | None:
        item = await super().pop_undo_item()
        self._save()
        return item

    async def pop_undo_group(self) -> list[ReversibleAction] | None:
        group = await super().pop_undo_group()
        self._save()
        return group

    async def pop_redo_item(self) -> ReversibleAction | None:
        item = await super().pop_redo_item()
        self._save()
        return item

    async def pop_redo_group(self) -> list[ReversibleAction] | None:
        group = await super().pop_redo_group()
        self._save()
        return group

    async def clear(self) -> None:
        await super().clear()
        self._save()
