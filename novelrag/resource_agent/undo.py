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
            case "apply":
                op = self.params.get("op", {})
                target = op.get("target", "unknown")
                if target == "property":
                    uri = op.get("resource_uri", "unknown")
                    keys = list(op.get("data", {}).keys())
                    if keys:
                        return f"Property update on {uri} (fields: {', '.join(keys)})"
                    return f"Property update on {uri}"
                elif target == "resource":
                    loc = op.get("location", {})
                    uri = loc.get("resource_uri", "unknown")
                    children_key = loc.get("children_key")
                    loc_label = f"{uri}/{children_key}" if children_key else uri
                    data = op.get("data")
                    start, end = op.get("start", 0), op.get("end", 0)
                    if data and end == start:
                        return f"Inserted {len(data)} resource(s) at {loc_label}"
                    elif not data and end > start:
                        return f"Removed {end - start} resource(s) from {loc_label}"
                    else:
                        return f"Spliced resources at {loc_label}"
                return "Applied operation on repository"
            case "update_relationships":
                src = self.params.get("source_uri", "unknown")
                tgt = self.params.get("target_uri", "unknown")
                rels = self.params.get("relationships", [])
                return f"Updated relationships between {src} and {tgt} ({len(rels)} relation(s))"
            case "add_aspect":
                name = self.params.get("name", "unknown")
                return f"Added aspect '{name}'"
            case "remove_aspect":
                name = self.params.get("name", "unknown")
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
