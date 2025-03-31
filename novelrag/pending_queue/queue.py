from .item import PendingUpdateItem


class PendingUpdateQueue:
    def __init__(self):
        self.queue: list[PendingUpdateItem] = []

    def push(self, item: PendingUpdateItem):
        self.queue.append(item)

    def lpush(self, item: PendingUpdateItem):
        self.queue.insert(0, item)

    def pop(self):
        if not self.queue:
            return None
        return self.queue.pop()

    def lpop(self):
        if not self.queue:
            return None
        return self.queue.pop(0)

    def clear(self):
        del self.queue
        self.queue = []
