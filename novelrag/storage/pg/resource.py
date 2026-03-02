from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import Element
from novelrag.resource.operation import Operation
from novelrag.resource.repository import ResourceRepository, SearchResult


class PostgresResourceRepository(ResourceRepository):
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def all_aspects(self) -> list[ResourceAspect]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute('SELECT id, name, uri, description, children_keys, metadata FROM resource_aspects')
                rows = await cur.fetchall()
                return [ResourceAspect(id=r['id'], name=r['name'], uri=r['uri'], description=r['description'], children_keys=r['children_keys'], aspect_meta=r['metadata']) for r in rows]

    async def get_aspect(self, name: str) -> ResourceAspect | None:
        pass

    async def add_aspect(self, name: str, metadata: dict[str, Any]) -> ResourceAspect:
        pass

    async def remove_aspect(self, name: str) -> ResourceAspect | None:
        pass

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = None) -> list[SearchResult]:
        """Search for resources using vector similarity.
        
        Args:
            query: The search query string
            aspect: Optional aspect to filter results
            limit: Maximum number of results to return
        """
        pass

    async def find_by_uri(self, resource_uri: str) -> list[ResourceAspect] | ResourceAspect | Element | None:
        """Find a resource by its URI in the repository.
        
        Args:
            resource_uri: The URI of the resource to find
        
        Returns:
            - list[ResourceAspect]: All aspects if resource_uri is '/'
            - ResourceAspect: Single aspect if resource_uri matches '/{aspect_name}'
            - Element: Element if found by URI
            - None: If no match is found
        """
        pass

    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository.
        
        Args:
            op: The operation to apply
        """
        pass

    async def update_relationships(self, source_uri: str, target_uri: str, relationships: list[str]) -> list[str]:
        """Update the relationships of a resource by its URI.
        Args:
            source_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relationships: A dictionary of relations to set for the resource
        Returns:
            List[str]: The old relationships before the update
        """
        pass
