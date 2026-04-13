"""In-process and local-file undo queue implementations."""

from novelrag.resource_agent.undo import ReversibleAction, UndoQueue


class MemoryUndoQueue(UndoQueue):
    """Undo/redo stacks kept entirely in memory."""

    def __init__(
        self,
        undo_stack: list[ReversibleAction] | None = None,
        redo_stack: list[ReversibleAction] | None = None,
        stack_size: int | None = 100,
    ) -> None:
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
    """Memory undo queue that persists to a local JSON file."""

    def __init__(
        self,
        path: str,
        undo_stack: list[ReversibleAction] | None = None,
        redo_stack: list[ReversibleAction] | None = None,
        stack_size: int | None = 100,
    ) -> None:
        self.path = path
        super().__init__(undo_stack, redo_stack, stack_size)

    @classmethod
    def load(cls, path: str, stack_size: int | None = 100) -> "LocalUndoQueue":
        import json
        import os

        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            undo_stack = [ReversibleAction(**item) for item in data.get("undo_stack", [])]
            redo_stack = [ReversibleAction(**item) for item in data.get("redo_stack", [])]
            return cls(path, undo_stack, redo_stack, stack_size)
        else:
            return cls(path, stack_size=stack_size)

    def _save(self) -> None:
        import json
        import os

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
