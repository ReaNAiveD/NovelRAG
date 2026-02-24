from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

# Maps string priority labels from LLM output to numeric values.
# Gaps between levels allow future intermediate priorities (e.g. 25 for "medium-high").
PRIORITY_MAP: dict[str, int] = {
    "high": 30,
    "normal": 20,
    "low": 10,
}
DEFAULT_PRIORITY = 20


def resolve_priority(value: int | str) -> int:
    """Convert a priority value to an integer.

    Accepts an int (passed through) or a string label mapped via PRIORITY_MAP.
    Falls back to DEFAULT_PRIORITY for unrecognised strings.
    """
    if isinstance(value, int):
        return value
    return PRIORITY_MAP.get(str(value).lower().strip(), DEFAULT_PRIORITY)


@dataclass
class BacklogEntry:
    """A backlog work item with structured metadata.

    Attributes:
        type: Category of the item (e.g. "dependency", "character_development").
        priority: Numeric priority (higher = more important).
        description: Human-readable description of the work to do.
        metadata: All additional fields from the LLM output that don't map
                  to the fixed attributes above (e.g. search_guidance,
                  rationale, target_resources, context_reference, …).
    """
    type: str
    priority: int
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- factory ---------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacklogEntry":
        """Build a BacklogEntry from a raw dict (typically LLM JSON output).

        Extracts the known keys (type, priority, description) and puts
        everything else into *metadata*.  If the dict already contains a
        ``metadata`` key (e.g. when reloading from persisted JSON), that
        value is used directly instead of being nested.
        """
        item_type = data.get("type", "other")
        priority = resolve_priority(data.get("priority", DEFAULT_PRIORITY))
        description = data.get("description", "")
        # If an explicit 'metadata' key is present, use it directly (reload path).
        # Otherwise, collect all extra keys as metadata (LLM output path).
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata = dict(data["metadata"])
        else:
            known_keys = {"type", "priority", "description"}
            metadata = {k: v for k, v in data.items() if k not in known_keys}
        return cls(type=item_type, priority=priority, description=description, metadata=metadata)

    # --- serialisation ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (suitable for JSON persistence)."""
        d: dict[str, Any] = {
            "type": self.type,
            "priority": self.priority,
            "description": self.description,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class Backlog(Protocol[T]):
    async def add_entry(self, entry: T) -> None:
        """Add an entry to the backlog."""
        ...

    async def get_entries(self) -> list[T]:
        """Get all entries from the backlog."""
        ...

    async def clear(self) -> None:
        """Clear the backlog."""
        ...

    async def get_top(self, n: int) -> list[T]:
        """Get the top n entries from the backlog."""
        ...

    async def pop_entry(self) -> T | None:
        """Get the entry with top priority from the backlog."""
        ...

    async def remove_entries(self, indices: list[int]) -> list[T]:
        """Remove entries at the given 0-based indices and return them."""
        ...

    def __len__(self) -> int:
        """Get the number of entries in the backlog."""
        ...
