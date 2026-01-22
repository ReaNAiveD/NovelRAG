from novelrag.resource_agent.backlog.memory import MemoryBacklog
from novelrag.resource_agent.backlog.types import PrioritizedBacklogEntry


class LocalBacklog(MemoryBacklog):
    def __init__(self, path: str, entries: list[PrioritizedBacklogEntry] | None = None) -> None:
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
        data = [PrioritizedBacklogEntry(**entry) for entry in data]

        return cls(path, data)
    
    def save(self) -> None:
        import os
        import json

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            data = [{"content": entry.content, "priority": entry.priority} for entry in self.entries]
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def add_entry(self, entry: PrioritizedBacklogEntry) -> None:
        super().add_entry(entry)
        self.save()
    
    def clear(self) -> None:
        super().clear()
        self.save()
    
    def pop_entry(self) -> PrioritizedBacklogEntry | None:
        entry = super().pop_entry()
        self.save()
        return entry
