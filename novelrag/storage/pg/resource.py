import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from psycopg import AsyncCursor
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import Element, load_elements
from novelrag.resource.operation import Operation, PropertyOperation, ResourceOperation
from novelrag.resource.repository import ResourceRepository, RemovedAspectResult, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class AspectRow:
    """An aspect together with its database row id."""
    aspect: ResourceAspect
    aspect_id: int


@dataclass
class ElementRow:
    """An element together with its database row id and parent aspect id."""
    element: Element
    aspect_id: int
    element_id: int


class PostgresResourceRepository(ResourceRepository):
    def __init__(self, workspace_id: int, pool: AsyncConnectionPool,
            embedder: Embeddings,):
        self.workspace_id = workspace_id
        self.pool = pool
        self.embedder = embedder

    # -- Row mapping helpers --------------------------------------------------

    @staticmethod
    def _row_to_aspect(row: dict) -> ResourceAspect:
        """Map a database row dict to a ``ResourceAspect``."""
        return ResourceAspect(
            name=row["name"],
            description=row["description"],
            children_keys=row["children_keys"],
            metadata=row["metadata"],
            root_element_names=row["root_element_names"],
        )

    @staticmethod
    def _row_to_element(row: dict, *, aspect_name: str | None = None) -> Element:
        """Map a database row dict to an ``Element``.

        Args:
            row: Dict row from the database.  Expected keys: ``name``,
                ``uri``, ``data``.  Optional: ``children_keys``,
                ``aspect_name``.
            aspect_name: Explicit aspect name override.  Falls back to
                ``row["aspect_name"]`` when *None*.
        """
        d: dict[str, Any] = {}
        if row.get('data'):
            d.update(row['data'])
        d['name'] = row['name']
        d['uri'] = row['uri']
        d['relationships'] = row.get('relationships', {})
        d['aspect'] = aspect_name or row['aspect_name']
        if row.get('children_keys') is not None:
            d['children_keys'] = row['children_keys']
        return Element.model_validate(d)


    @staticmethod
    def _build_undo_dict(
        uri: str,
        row_by_uri: dict[str, dict],
        children_keys: list[str],
    ) -> dict[str, Any] | None:
        """Recursively reconstruct a nested element dict from flat deleted rows.

        Returns a dict suitable for ``load_elements`` / ``create_undo``, or
        *None* when the URI is not present in *row_by_uri*.
        """
        row = row_by_uri.get(uri)
        if row is None:
            return None
        result: dict[str, Any] = {'id': row['name']}
        result['relationships'] = row.get('relationships', {})
        if row['data']:
            result.update(row['data'])
        for key in children_keys:
            child_names = row['data'].get(key, []) if row['data'] else []
            nested = []
            for child_name in child_names:
                child = PostgresResourceRepository._build_undo_dict(
                    f'{uri}/{child_name}', row_by_uri, children_keys,
                )
                if child is not None:
                    nested.append(child)
            result[key] = nested
        return result

    async def _delete_and_build_undo(
        self,
        cur: AsyncCursor[DictRow],
        aspect_id: int,
        undo_uris: list[str],
        children_keys: list[str],
    ) -> list[dict[str, Any]]:
        """Delete elements by URI (including descendants) and return undo data."""
        if not undo_uris:
            return []
        await cur.execute(
            "DELETE FROM resource_elements "
            "WHERE workspace_id = %(ws)s AND aspect_id = %(aspect_id)s AND ("
            "  uri = ANY(%(uris)s) OR "
            "  EXISTS (SELECT 1 FROM unnest(%(prefixes)s) p WHERE starts_with(uri, p))"
            ") RETURNING name, uri, data, relationships",
            {
                'ws': self.workspace_id,
                'aspect_id': aspect_id,
                'uris': undo_uris,
                'prefixes': [uri + '/' for uri in undo_uris],
            },
        )
        deleted_rows = await cur.fetchall()
        row_by_uri: dict[str, dict] = {r['uri']: r for r in deleted_rows}
        undo_data: list[dict[str, Any]] = []
        for uri in undo_uris:
            nested = self._build_undo_dict(uri, row_by_uri, children_keys)
            if nested is not None:
                undo_data.append(nested)
        return undo_data

    async def _insert_elements(
        self,
        cur: AsyncCursor[DictRow],
        aspect_id: int,
        elements: list[Element],
    ) -> None:
        """Batch-insert elements with their embeddings."""
        if not elements:
            return
        texts = [e.element_str() for e in elements]
        embeddings = await self.embedder.aembed_documents(texts)
        await cur.executemany(
            "INSERT INTO resource_elements (workspace_id, aspect_id, name, uri, relationships, data, embedding) "
            "VALUES (%(ws)s, %(aspect_id)s, %(name)s, %(uri)s, %(relationships)s, %(data)s, %(embedding)s)",
            [
                {
                    'ws': self.workspace_id,
                    'aspect_id': aspect_id,
                    'name': e.id,
                    'uri': e.uri,
                    'relationships': e.relationships,
                    'data': e.model_extra,
                    'embedding': embedding,
                }
                for e, embedding in zip(elements, embeddings)
            ],
        )

    async def _update_element_data(
        self,
        conn,
        element: Element,
        props: dict[str, Any],
    ) -> dict[str, Any]:
        """Update element properties, recompute embedding, and persist.

        Returns the previous property values (undo dict).
        """
        undo = element.update(props)
        embedding = await self.embedder.aembed_query(element.element_str())
        await conn.execute(
            "UPDATE resource_elements SET name = %(name)s, data = %(data)s, embedding = %(embedding)s "
            "WHERE workspace_id = %(ws)s AND uri = %(uri)s",
            {
                'ws': self.workspace_id,
                'name': element.id,
                'data': element.model_extra,
                'embedding': embedding,
                'uri': element.uri,
            }
        )
        return undo

    async def _update_aspect_data(
        self,
        conn,
        aspect: ResourceAspect,
        props: dict[str, Any],
    ) -> dict[str, Any]:
        """Update aspect properties and persist.

        Returns the previous property values (undo dict).
        """
        undo = aspect.update(props)
        await conn.execute(
            "UPDATE resource_aspects SET description = %(description)s, children_keys = %(children_keys)s, metadata = %(metadata)s "
            "WHERE workspace_id = %(ws)s AND name = %(name)s",
            {
                'ws': self.workspace_id,
                'name': aspect.name,
                'description': aspect.description,
                'children_keys': aspect.children_keys,
                'metadata': aspect.metadata,
            }
        )
        return undo

    async def _splice_children(
        self,
        cur: AsyncCursor[DictRow],
        aspect_id: int,
        parent_uri: str,
        aspect_name: str,
        children_keys: list[str],
        current_names: list[str],
        op: ResourceOperation,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Splice children: load new elements, delete replaced, insert new.

        Returns ``(new_children_names, undo_data)``.
        """
        elements, descendants = load_elements(op.data or [], parent_uri, aspect_name, children_keys)
        undo_uris = [f'{parent_uri}/{name}' for name in current_names[op.start:op.end]]
        new_names = current_names[:op.start] + [e.id for e in elements] + current_names[op.end:]
        undo_data = await self._delete_and_build_undo(cur, aspect_id, undo_uris, children_keys)
        await self._insert_elements(cur, aspect_id, elements + descendants)
        return new_names, undo_data


    async def _all_aspect_names(self, cur: AsyncCursor[DictRow]) -> list[str]:
        """Return every aspect name in the workspace."""
        await cur.execute(
            "SELECT name FROM resource_aspects WHERE workspace_id = %(ws)s",
            {'ws': self.workspace_id},
        )
        rows = await cur.fetchall()
        if not rows:
            return []
        return [r['name'] for r in rows]

    async def _all_aspects(self, cur: AsyncCursor[DictRow]) -> list[AspectRow]:
        """Return every ``ResourceAspect`` in the workspace."""
        await cur.execute(
            "SELECT id, name, uri, description, children_keys, metadata, root_element_names "
            "FROM resource_aspects WHERE workspace_id = %(ws)s",
            {'ws': self.workspace_id},
        )
        rows = await cur.fetchall()
        return [AspectRow(aspect=self._row_to_aspect(r), aspect_id=r['id']) for r in rows]

    async def _get_aspect(self, cur: AsyncCursor[DictRow], name: str) -> AspectRow | None:
        """Fetch a single aspect by name, or *None*."""
        await cur.execute(
            "SELECT id, name, uri, description, children_keys, metadata, root_element_names "
            "FROM resource_aspects WHERE workspace_id = %(ws)s AND name = %(name)s",
            {'ws': self.workspace_id, 'name': name},
        )
        row = await cur.fetchone()
        return AspectRow(aspect=self._row_to_aspect(row), aspect_id=row['id']) if row else None

    async def _find_element_by_uri(self, cur: AsyncCursor[DictRow], uri: str) -> ElementRow | None:
        """Fetch an element by its full URI, or *None*."""
        await cur.execute(
            "SELECT e.id, e.name, e.uri, e.relationships, e.data, a.id AS aspect_id, a.children_keys, a.name AS aspect_name "
            "FROM resource_elements e "
            "JOIN resource_aspects a ON e.aspect_id = a.id "
            "WHERE e.workspace_id = %(ws)s AND a.workspace_id = %(ws)s AND e.uri = %(uri)s",
            {'ws': self.workspace_id, 'uri': uri},
        )
        row = await cur.fetchone()
        return ElementRow(element=self._row_to_element(row), aspect_id=row['aspect_id'], element_id=row['id']) if row else None

    async def _iter_elements_of_aspect(self, cur: AsyncCursor[DictRow], aspect_name: str) -> list[ElementRow]:
        """Return every element belonging to *aspect_name*."""
        await cur.execute(
            "SELECT e.id, e.name, e.uri, e.relationships, e.data, a.id AS aspect_id, a.children_keys, a.name AS aspect_name "
            "FROM resource_elements e "
            "JOIN resource_aspects a ON e.aspect_id = a.id "
            "WHERE e.workspace_id = %(ws)s AND a.workspace_id = %(ws)s AND a.name = %(aspect_name)s",
            {'ws': self.workspace_id, 'aspect_name': aspect_name},
        )
        rows = await cur.fetchall()
        return [ElementRow(element=self._row_to_element(r, aspect_name=aspect_name), aspect_id=r['aspect_id'], element_id=r['id']) for r in rows]

    async def _find_by_uri(
        self, cur: AsyncCursor[DictRow], resource_uri: str,
    ) -> list[str] | AspectRow | ElementRow | None:
        """Dispatch a URI lookup without acquiring a new connection."""
        if not resource_uri.startswith('/'):
            raise ValueError("Invalid URI: must start with '/'")
        if resource_uri == '/':
            return await self._all_aspect_names(cur)
        elif resource_uri.count('/') == 1:
            return await self._get_aspect(cur, resource_uri[1:])
        else:
            return await self._find_element_by_uri(cur, resource_uri)

    async def _vector_search(
        self, cur: AsyncCursor[DictRow], query_embedding,
        *, aspect: str | None = None, limit: int | None = None,
    ) -> list[SearchResult]:
        """Execute a cosine-similarity search against element embeddings."""
        # Use <=> operator for cosine similarity search;
        # the index on the embedding column is built with vector_cosine_ops.
        sql = (
            "SELECT e.name, e.uri, e.relationships, e.data, a.name AS aspect_name, a.children_keys, "
            "(e.embedding <=> %(query_embedding)s) AS _distance "
            "FROM resource_elements e "
            "JOIN resource_aspects a ON e.aspect_id = a.id "
            "WHERE e.workspace_id = %(ws)s AND a.workspace_id = %(ws)s "
        )
        params: dict[str, Any] = {
            'ws': self.workspace_id,
            'query_embedding': query_embedding,
        }
        if aspect:
            sql += "AND a.name = %(aspect)s "
            params['aspect'] = aspect
        sql += "ORDER BY e.embedding <=> %(query_embedding)s "
        if limit:
            sql += "LIMIT %(limit)s"
            params['limit'] = limit
        await cur.execute(sql, params)
        rows = await cur.fetchall()
        return [
            SearchResult(
                distance=row['_distance'],
                element=self._row_to_element(row),
            )
            for row in rows
        ]

    # -- Public interface (ResourceRepository) --------------------------------

    async def all_aspects(self) -> list[ResourceAspect]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                return [row.aspect for row in await self._all_aspects(cur)]

    async def get_aspect(self, name: str) -> ResourceAspect | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                result = await self._get_aspect(cur, name)
                return result.aspect if result else None

    async def add_aspect(self, name: str, metadata: dict[str, Any], elements: list[dict] | None = None) -> ResourceAspect:
        description = metadata.get('description', '')
        children_keys = metadata.get('children_keys', [])
        root_element_names = metadata.get('root_element_names', [])
        aspect = ResourceAspect(name=name, description=description, children_keys=children_keys, metadata=metadata, root_element_names=root_element_names)
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT 1 FROM resource_aspects WHERE workspace_id = %(ws)s AND name = %(name)s",
                        {'ws': self.workspace_id, 'name': name}
                    )
                    if await cur.fetchone():
                        raise ValueError(f"Aspect with name '{name}' already exists in workspace {self.workspace_id}.")
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "INSERT INTO resource_aspects (workspace_id, name, uri, description, children_keys, metadata, root_element_names) "
                        "VALUES (%(ws)s, %(name)s, %(uri)s, %(description)s, %(children_keys)s, %(metadata)s, %(root_element_names)s)"
                        "RETURNING id",
                        {
                            'ws': self.workspace_id,
                            'name': aspect.name,
                            'uri': f'/{name}',
                            'description': aspect.description,
                            'children_keys': aspect.children_keys,
                            'metadata': aspect.metadata,
                            'root_element_names': aspect.root_element_names,
                        }
                    )
                    inserted = await cur.fetchone()
                    if not inserted:
                        raise RuntimeError(f"Failed to insert aspect '{name}' into workspace {self.workspace_id}.")
                    aspect_id = inserted['id']
                    if elements:
                        await cur.executemany(
                            "INSERT INTO resource_elements (workspace_id, aspect_id, name, uri, relationships, data, embedding) "
                            "VALUES (%(ws)s, %(aspect_id)s, %(name)s, %(uri)s, %(relationships)s, %(data)s, %(embedding)s)",
                            [
                                {
                                    'ws': self.workspace_id,
                                    'aspect_id': aspect_id,
                                    'name': elem['name'],
                                    'uri': elem['uri'],
                                    'relationships': elem.get('relationships', {}),
                                    'data': elem['data'],
                                    'embedding': elem['embedding'],
                                }
                                for elem in elements
                            ],
                        )
        return aspect

    async def remove_aspect(self, name: str) -> RemovedAspectResult | None:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "DELETE FROM resource_aspects WHERE workspace_id = %(ws)s AND name = %(name)s "
                        "RETURNING id, name, description, children_keys, metadata, root_element_names",
                        {'ws': self.workspace_id, 'name': name}
                    )
                    row = await cur.fetchone()
                    if row:
                        aspect = self._row_to_aspect(row)
                        # Elements are not automatically deleted due to the lack of CASCADE
                        # so we need to delete them manually and return them for undo purposes
                        await cur.execute(
                            "DELETE FROM resource_elements WHERE workspace_id = %(ws)s AND aspect_id = %(aspect_id)s "
                            "RETURNING name, uri, relationships, data, embedding",
                            {'ws': self.workspace_id, 'aspect_id': row['id']},
                        )
                        elements = await cur.fetchall()
                        elements = [{'name': e['name'], 'uri': e['uri'], 'relationships': e.get('relationships', {}), 'data': e['data'], 'embedding': e['embedding']} for e in elements]
                        return RemovedAspectResult(aspect=aspect, elements=elements, restore_context={})
                    return None

    async def iter_elements(self, aspect_name: str) -> list[Element]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                return [row.element for row in await self._iter_elements_of_aspect(cur, aspect_name)]

    async def find_by_uri(self, resource_uri: str) -> list[str] | ResourceAspect | Element | None:
        """Find a resource by its URI in the repository.
        
        Args:
            resource_uri: The URI of the resource to find
        
        Returns:
            - list[str]: All aspects if resource_uri is '/'
            - ResourceAspect: Single aspect if resource_uri matches '/{aspect_name}'
            - Element: Element if found by URI
            - None: If no match is found
        """
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                result = await self._find_by_uri(cur, resource_uri)
                if isinstance(result, AspectRow):
                    return result.aspect
                elif isinstance(result, ElementRow):
                    return result.element
                return result

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = None) -> list[SearchResult]:
        """Search for resources using vector similarity.
        
        Args:
            query: The search query string
            aspect: Optional aspect to filter results
            limit: Maximum number of results to return
        """
        query_embedding = await self.embedder.aembed_query(query)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                return await self._vector_search(cur, query_embedding, aspect=aspect, limit=limit)

    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository."""
        if isinstance(op, PropertyOperation):
            return await self._apply_property_op(op)
        elif isinstance(op, ResourceOperation):
            return await self._apply_resource_op(op)
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")

    async def _apply_property_op(self, op: PropertyOperation) -> Operation:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    resource = await self._find_by_uri(cur, op.resource_uri)
                if isinstance(resource, list):
                    raise ValueError("Cannot apply PropertyOperation to aspect list URI '/'")
                elif isinstance(resource, AspectRow):
                    undo = await self._update_aspect_data(conn, resource.aspect, op.data)
                    return op.create_undo(undo)
                elif isinstance(resource, ElementRow):
                    undo = await self._update_element_data(conn, resource.element, op.data)
                    return op.create_undo(undo)
                else:
                    raise ValueError(f"Resource with URI '{op.resource_uri}' not found for PropertyOperation.")

    async def _apply_resource_op(self, op: ResourceOperation) -> Operation:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    target = await self._find_by_uri(cur, op.location.resource_uri)
                    if isinstance(target, list):
                        raise ValueError("Cannot apply ResourceOperation to aspect list URI '/'")
                    elif isinstance(target, AspectRow):
                        aspect = target.aspect
                        if op.location.children_key is not None:
                            logger.warning(f"ResourceOperation location for aspect '{aspect.name}' should not specify children_key, but got '{op.location.children_key}'. Ignoring children_key.")
                        new_names, undo_data = await self._splice_children(
                            cur, target.aspect_id, op.location.resource_uri,
                            aspect.name, aspect.children_keys or [],
                            aspect.root_element_names, op,
                        )
                        aspect.root_element_names = new_names
                        await conn.execute(
                            "UPDATE resource_aspects SET root_element_names = %(root_element_names)s "
                            "WHERE workspace_id = %(ws)s AND name = %(name)s",
                            {
                                'ws': self.workspace_id,
                                'name': aspect.name,
                                'root_element_names': aspect.root_element_names,
                            }
                        )
                        return op.create_undo(undo_data)
                    elif isinstance(target, ElementRow):
                        element = target.element
                        if op.location.children_key is None:
                            raise ValueError(f"ResourceOperation location for element '{element.uri}' must specify children_key.")
                        new_names, undo_data = await self._splice_children(
                            cur, target.aspect_id, element.uri,
                            element.aspect, element.children_keys or [],
                            element.children_names_of(op.location.children_key), op,
                        )
                        element.set_children_names(op.location.children_key, new_names)
                        await conn.execute(
                            "UPDATE resource_elements SET data = %(data)s "
                            "WHERE workspace_id = %(ws)s AND uri = %(uri)s",
                            {
                                'ws': self.workspace_id,
                                'uri': element.uri,
                                'data': element.model_extra,
                            })
                        return op.create_undo(undo_data)
                    else:
                        raise ValueError(f"Resource with URI '{op.location.resource_uri}' not found for ResourceOperation.")


    async def update_relationships(self, source_uri: str, target_uri: str, relationships: list[str]) -> list[str]:
        """Update the relationships of a resource by its URI.
        Args:
            source_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relationships: A dictionary of relations to set for the resource
        Returns:
            List[str]: The old relationships before the update
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    src_element = await self._find_element_by_uri(cur, source_uri)
                    if not src_element:
                        raise ValueError(f"Source element with URI '{source_uri}' not found for updating relationships.")
                    tgt_element = await self._find_element_by_uri(cur, target_uri)
                    if not tgt_element:
                        raise ValueError(f"Target element with URI '{target_uri}' not found for updating relationships.")
                    undo = src_element.element.update_relationships(target_uri, relationships)
                    await conn.execute(
                        "UPDATE resource_elements SET relationships = %(relationships)s "
                        "WHERE workspace_id = %(ws)s AND uri = %(uri)s",
                        {
                            'ws': self.workspace_id,
                            'uri': src_element.element.uri,
                            'relationships': src_element.element.relationships,
                        }
                    )
                    return undo
