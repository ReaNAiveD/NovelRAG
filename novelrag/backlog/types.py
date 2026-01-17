from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")


@dataclass
class PrioritizedBacklogEntry:
    content: str
    priority: int


class Backlog(Protocol[T]):
    def add_entry(self, entry: T) -> None:
        """Add an entry to the backlog."""
        ...

    def get_entries(self) -> list[T]:
        """Get all entries from the backlog."""
        ...

    def clear(self) -> None:
        """Clear the backlog."""
        ...

    def get_top(self, n: int) -> list[T]:
        """Get the top n entries from the backlog."""
        ...

    def pop_entry(self) -> T | None:
        """Get the entry with top priority from the backlog."""
        ...

    def __len__(self) -> int:
        """Get the number of entries in the backlog."""
        ...
