from dataclasses import dataclass

from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource import Operation


@dataclass
class UndoItem:
    ops: list[Operation]
    redo: PendingUpdateItem | None


class UndoQueue:
    def __init__(self):
        self.queue: list[UndoItem] = []

    def push(self, ops: list[Operation], redo: PendingUpdateItem | None = None):
        self.queue.append(UndoItem(ops=ops, redo=redo))

    def pop(self):
        if not self.queue:
            return None
        return self.queue.pop()
