import lancedb
from lancedb import AsyncConnection, AsyncTable
from lancedb.index import IvfPq
from lancedb.pydantic import LanceModel, Vector


class EmbeddingSearch(LanceModel):
    vector: Vector(3072)
    element_id: str
    aspect: str


class LanceDBStore:
    def __init__(self, connection: AsyncConnection, table: AsyncTable):
        self.connection = connection
        self.table = table

    @classmethod
    async def create(cls, uri: str, table_name: str, init_data: list | None = None, overwrite=False):
        if not init_data:
            init_data = None
        connection = await lancedb.connect_async(uri)
        if overwrite:
            table = await connection.create_table(table_name, schema=EmbeddingSearch, data=init_data, mode="overwrite")
        else:
            table = await connection.create_table(table_name, schema=EmbeddingSearch, data=init_data, exist_ok=True)
        if init_data:
            await table.create_index('vector', replace=True, config=IvfPq())
        store = cls(connection=connection, table=table)
        return store

    async def vector_search(self, vector: list[float], *, aspect: str | None = None, limit: int | None =20) -> list[dict]:
        qry = self.table.vector_search(vector)
        if aspect:
            qry = qry.where(f'aspect = "{aspect}"')
        if limit is not None:
            qry = qry.limit(limit)
        return await qry.to_list()

    async def add(self, ele_id: str, vector: list[float], aspect: str):
        return await self.table.add([EmbeddingSearch(
            vector=vector,
            element_id=ele_id,
            aspect=aspect,
        )])

    async def update_vector(self, ele_id: str, vector: list[float]):
        return await self.table.update(
            updates={'vector': vector},
            where=f'element_id = "{ele_id}"',
        )

    async def delete(self, ele_id: str):
        return await self.table.delete(where=f'element_id = "{ele_id}"')
