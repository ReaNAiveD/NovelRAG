from novelrag.resource_agent.backlog.memory import MemoryBacklog
from novelrag.resource_agent.backlog.types import BacklogEntry


class LocalBacklog(MemoryBacklog):
    def __init__(self, path: str, entries: list[BacklogEntry] | None = None) -> None:
        self.path = path
        super().__init__(entries)
    
    @classmethod
    def load(cls, path: str) -> "LocalBacklog":
        import os
        import json

        if not os.path.exists(path):
            return cls(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        entries = []
        for entry in data:
            # Support both new format (type/priority/description/metadata)
            # and legacy format (content/priority)
            if "description" in entry or "type" in entry:
                entries.append(BacklogEntry.from_dict(entry))
            else:
                # Legacy: {"content": "...", "priority": ...}
                entries.append(BacklogEntry(
                    type="other",
                    priority=entry.get("priority", 20) if isinstance(entry.get("priority"), int) else 20,
                    description=entry.get("content", ""),
                ))

        return cls(path, entries)
    
    def save(self) -> None:
        import os
        import json

        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            data = [entry.to_dict() for entry in self.entries]
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def add_entry(self, entry: BacklogEntry) -> None:
        super().add_entry(entry)
        self.save()
    
    def clear(self) -> None:
        super().clear()
        self.save()
    
    def pop_entry(self) -> BacklogEntry | None:
        entry = super().pop_entry()
        self.save()
        return entry

    def remove_entries(self, indices: list[int]) -> list[BacklogEntry]:
        removed = super().remove_entries(indices)
        if removed:
            self.save()
        return removed
