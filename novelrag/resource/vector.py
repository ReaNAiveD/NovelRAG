import abc
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import lancedb
from lancedb import AsyncConnection, AsyncTable
from lancedb.pydantic import LanceModel, Vector

from novelrag.llm import EmbeddingLLM
from novelrag.resource import Element


class Hasher(abc.ABC):
    """Abstract base class for hash generation utilities."""

    def hash(self, target: str) -> str:
        """Generate hash from a string."""
        raise NotImplementedError()

    def hash_obj(self, obj: dict) -> str:
        """Generate hash from a dictionary object."""
        serialized_data = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return self.hash(serialized_data)


class Md5Hasher(Hasher):
    """MD5 implementation of the Hasher interface."""

    def hash(self, target: str) -> str:
        """Generate MD5 hash from input string."""
        return hashlib.md5(target.encode('utf-8')).hexdigest()


class EmbeddingSearch(LanceModel):
    """LanceDB schema for storing embedding vectors and metadata."""
    vector: Vector(3072)
    hash: str
    element_id: str
    aspect: str


@dataclass(init=False)
class VectorSearchResult:
    def __init__(self, item: dict):
        validated_item = EmbeddingSearch.model_validate(item)
        self.vector = validated_item.vector
        self.hash = validated_item.hash
        self.element_id = validated_item.element_id
        self.aspect = validated_item.aspect
        self.distance = item['_distance']


class LanceDBStore:
    """Vector store implementation using LanceDB for embedding storage and retrieval.

    Attributes:
        connection: Async connection to LanceDB
        table: Active table instance for operations
        embedder: Embedding generator for text data
        hasher: Hash generator for data versioning
    """

    def __init__(self, connection: AsyncConnection, table: AsyncTable,
                 embedder: EmbeddingLLM, hasher: Hasher):
        """Initialize with pre-configured dependencies."""
        self.connection = connection
        self.table = table
        self.embedder = embedder
        self.hasher = hasher

    @classmethod
    async def create(cls, uri: str, table_name: str, embedder: EmbeddingLLM,
                     hasher: Optional[Hasher] = None) -> 'LanceDBStore':
        """Factory method to create and initialize a store instance.

        Args:
            uri: Database connection URI
            table_name: Name of the table to create/use
            embedder: Text embedding generator
            hasher: Optional hash generator (defaults to MD5)

        Returns:
            Configured LanceDBStore instance
        """
        hasher = hasher or Md5Hasher()
        connection = await lancedb.connect_async(uri)
        table = await connection.create_table(
            table_name,
            schema=EmbeddingSearch,
            exist_ok=True
        )
        return cls(connection, table, embedder, hasher)

    async def vector_search(self, vector: list[float], *,
                            aspect: Optional[str] = None,
                            limit: Optional[int] = 20) -> list[VectorSearchResult]:
        """Search for similar vectors in the store.

        Args:
            vector: Query vector for similarity search
            aspect: Optional filter for specific aspect type
            limit: Maximum number of results to return

        Returns:
            List of matching EmbeddingSearch results
        """
        query = self.table.vector_search(vector)
        if aspect:
            query = query.where(f'aspect = "{aspect}"')
        if limit is not None:
            query = query.limit(limit)

        results = await query.to_list()
        return [VectorSearchResult(item) for item in results]

    async def get(self, element_id: str) -> Optional[EmbeddingSearch]:
        """Retrieve a single element by its unique ID.

        Args:
            element_id: Unique identifier of the element

        Returns:
            EmbeddingSearch instance if found, None otherwise
        """
        results = await self.table.query() \
            .where(f'element_id = "{element_id}"') \
            .limit(1) \
            .to_list()

        return EmbeddingSearch.model_validate(results[0]) if results else None

    async def batch_add(self, elements: list[Element]):
        """Batch insert multiple elements with empty table optimization.

        Args:
            elements: List of Element objects to add
        """
        if await self._is_table_empty():
            lines = [await self._create_line(ele) for ele in elements]
            return await self.table.add(lines)

        for element in elements:
            await self.add(element)

    async def add(self, element: Element, *, unchecked: bool = False):
        """Add a single element with optional existence check.

        Args:
            element: Element to add
            unchecked: Skip existence verification when True
        """
        return await self._upsert_element(
            element.id,
            element.aspect,
            element.element_dict(),
            unchecked=unchecked
        )

    async def update(self, element: Element):
        """Update an existing element's data.

        Args:
            element: Element with updated data
        """
        return await self._update_element(
            element.id,
            element.aspect,
            element.element_dict()
        )

    async def delete(self, element_id: str):
        """Remove an element by its ID.

        Args:
            element_id: Unique identifier of the element to remove
        """
        return await self.table.delete(where=f'element_id = "{element_id}"')

    # Implementation details below
    async def _is_table_empty(self) -> bool:
        """Check if the table contains any records."""
        return await self.table.count_rows() == 0

    async def _create_line(self, element: Element) -> EmbeddingSearch:
        """Create EmbeddingSearch instance from an Element."""
        serialized_data = json.dumps(
            element.element_dict(),
            ensure_ascii=False,
            sort_keys=True
        )
        hash_str = self.hasher.hash(serialized_data)
        embedding = await self.embedder.embedding(serialized_data, dimensions=3072)

        return EmbeddingSearch(
            vector=embedding[0],
            element_id=element.id,
            aspect=element.aspect,
            hash=hash_str
        )

    async def _upsert_element(self, element_id: str, aspect: str, data: dict,
                              unchecked: bool = False):
        """Internal method to handle element insertion/update."""
        serialized_data = json.dumps(data, ensure_ascii=False, sort_keys=True)
        hash_str = self.hasher.hash(serialized_data)
        existing = None

        if not unchecked:
            existing = await self.get(element_id)
            if existing and existing.hash == hash_str:
                return

        embedding = await self.embedder.embedding(serialized_data, dimensions=3072)
        vector = embedding[0]

        if existing:
            return await self._update_record(element_id, vector, hash_str)

        return await self.table.add([EmbeddingSearch(
            vector=vector,
            element_id=element_id,
            aspect=aspect,
            hash=hash_str
        )])

    async def _update_element(self, element_id: str, aspect: str, data: dict):
        """Internal method to handle element updates."""
        serialized_data = json.dumps(data, ensure_ascii=False, sort_keys=True)
        hash_str = self.hasher.hash(serialized_data)
        embedding = await self.embedder.embedding(serialized_data, dimensions=3072)

        return await self._update_record(
            element_id,
            embedding[0],
            hash_str,
            aspect
        )

    async def _update_record(self, element_id: str, vector: list[float],
                             hash_str: str, aspect: Optional[str] = None):
        """Execute update operation on the database table."""
        updates = {'vector': vector, 'hash': hash_str}
        if aspect:
            updates['aspect'] = aspect

        return await self.table.update(
            updates=updates,
            where=f'element_id = "{element_id}"'
        )
