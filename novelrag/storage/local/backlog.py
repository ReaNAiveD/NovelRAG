"""In-process and local-file backlog implementations."""

from novelrag.resource_agent.backlog import Backlog, BacklogEntry


class MemoryBacklog(Backlog[BacklogEntry]):
    """Priority-sorted backlog kept entirely in memory."""

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


class LocalBacklog(MemoryBacklog):
    """Memory backlog that persists to a local JSON file."""

    def __init__(self, path: str, entries: list[BacklogEntry] | None = None) -> None:
        self.path = path
        super().__init__(entries)

    @classmethod
    def load(cls, path: str) -> "LocalBacklog":
        import json
        import os

        if not os.path.exists(path):
            return cls(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Invalid backlog file format at {path!r}: expected a list of entries, got {type(data).__name__}"
            )
        entries = []
        for entry in data:
            # Support both new format (type/priority/description/metadata)
            # and legacy format (content/priority)
            if "description" in entry or "type" in entry:
                entries.append(BacklogEntry.from_dict(entry))
            else:
                # Legacy: {"content": "...", "priority": ...}
                entries.append(
                    BacklogEntry(
                        type="other",
                        priority=entry.get("priority", 20) if isinstance(entry.get("priority"), int) else 20,
                        description=entry.get("content", ""),
                    )
                )

        return cls(path, entries)

    def save(self) -> None:
        import json
        import os

        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            data = [entry.to_dict() for entry in self.entries]
            json.dump(data, f, indent=4, ensure_ascii=False)

    async def add_entry(self, entry: BacklogEntry) -> None:
        await super().add_entry(entry)
        self.save()

    async def clear(self) -> None:
        await super().clear()
        self.save()

    async def pop_entry(self) -> BacklogEntry | None:
        entry = await super().pop_entry()
        self.save()
        return entry

    async def remove_entries(self, indices: list[int]) -> list[BacklogEntry]:
        removed = await super().remove_entries(indices)
        if removed:
            self.save()
        return removed
