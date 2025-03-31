from dataclasses import dataclass

from novelrag.resource import Operation


@dataclass
class PendingUpdateItem:
    ops: list[Operation]
    generated: dict[str, str] | None = None
