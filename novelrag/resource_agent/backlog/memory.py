from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry


class MemoryBacklog(Backlog[BacklogEntry]):
    def __init__(self, entries: list[BacklogEntry] | None = None) -> None:
        self.entries: list[BacklogEntry] = entries if entries is not None else []
        self._sort()

    def _sort(self) -> None:
        """Keep entries in descending priority order."""
        self.entries.sort(key=lambda e: e.priority, reverse=True)

    async def add_entry(self, entry: BacklogEntry) -> None:
        self.entries.append(entry)
        self._sort()

    async def get_entries(self) -> list[BacklogEntry]:
        return self.entries

    async def clear(self) -> None:
        self.entries = []
    
    async def get_top(self, n: int) -> list[BacklogEntry]:
        return self.entries[:n]

    async def pop_entry(self) -> BacklogEntry | None:
        if not self.entries:
            return None
        return self.entries.pop(0)

    async def remove_entries(self, indices: list[int]) -> list[BacklogEntry]:
        """Remove entries at the given 0-based indices and return them.

        Indices refer to the current (sorted) order of ``self.entries``.
        Out-of-range indices are silently ignored.
        """
        valid = sorted(set(idx for idx in indices if 0 <= idx < len(self.entries)), reverse=True)
        removed = []
        for idx in valid:
            removed.append(self.entries.pop(idx))
        removed.reverse()  # return in ascending-index order
        return removed

    def __len__(self) -> int:
        return len(self.entries)
