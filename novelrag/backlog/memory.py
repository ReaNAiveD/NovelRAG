from novelrag.backlog.types import Backlog, PrioritizedBacklogEntry


class MemoryBacklog(Backlog[PrioritizedBacklogEntry]):
    def __init__(self, entries: list[PrioritizedBacklogEntry] | None = None) -> None:
        self.entries: list[PrioritizedBacklogEntry] = entries if entries is not None else []
    
    def add_entry(self, entry: PrioritizedBacklogEntry) -> None:
        self.entries.append(entry)

    def get_entries(self) -> list[PrioritizedBacklogEntry]:
        return self.entries

    def clear(self) -> None:
        self.entries = []
    
    def get_top(self, n: int) -> list[PrioritizedBacklogEntry]:
        sorted_entries = sorted(self.entries, key=lambda e: e.priority, reverse=True)
        return sorted_entries[:n]

    def pop_entry(self) -> PrioritizedBacklogEntry | None:
        if not self.entries:
            return None
        top_entry = max(self.entries, key=lambda e: e.priority)
        self.entries.remove(top_entry)
    
    def __len__(self) -> int:
        return len(self.entries)
