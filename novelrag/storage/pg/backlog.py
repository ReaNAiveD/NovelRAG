from asyncpg import Pool
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry


class PostgresBacklog(Backlog[BacklogEntry]):
    def __init__(self, pool: Pool):
        self.pool = pool

    async def add_entry(self, entry: BacklogEntry) -> None:
        pass

    async def get_entries(self) -> list[BacklogEntry]:
        pass

    async def clear(self) -> None:
        pass

    async def get_top(self, n: int) -> list[BacklogEntry]:
        pass

    async def pop_entry(self) -> BacklogEntry | None:
        pass

    async def remove_entries(self, indices: list[int]) -> list[BacklogEntry]:
        pass
