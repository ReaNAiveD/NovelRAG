import abc
import hashlib
import json
from dataclasses import dataclass
import logging
from typing import Optional

import lancedb
from lancedb import AsyncConnection, AsyncTable
from lancedb.pydantic import LanceModel, Vector

from langchain_core.embeddings import Embeddings
from novelrag.resource.element import Element

logger = logging.getLogger(__name__)


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
    resource_uri: str
    aspect: str


@dataclass(init=False)
class VectorSearchResult:
    def __init__(self, item: dict):
        validated_item = EmbeddingSearch.model_validate(item)
        self.vector = validated_item.vector
        self.hash = validated_item.hash
        self.resource_uri = validated_item.resource_uri
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
                 embedder: Embeddings, hasher: Hasher):
        """Initialize with pre-configured dependencies."""
        self.connection = connection
        self.table = table
        self.embedder = embedder
        self.hasher = hasher

    @classmethod
    async def create(cls, uri: str, table_name: str, embedder: Embeddings,
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

    async def get(self, resource_uri: str) -> Optional[EmbeddingSearch]:
        """Retrieve a single resource by its unique ID.

        Args:
            resource_uri: Unique identifier of the resource

        Returns:
            EmbeddingSearch instance if found, None otherwise
        """
        results = await self.table.query() \
            .where(f'resource_uri = "{resource_uri}"') \
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
            if lines:
                await self.table.add(lines)

        for element in elements:
            await self.add(element)

    async def add(self, element: Element, *, unchecked: bool = False):
        """Add a single element with optional existence check.

        Args:
            element: Element to add
            unchecked: Skip existence verification when True
        """
        return await self._upsert_element(
            element.uri,
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
            element.uri,
            element.aspect,
            element.element_dict()
        )

    async def delete(self, resource_uri: str):
        """Remove a resource by its ID.

        Args:
            resource_uri: Unique identifier of the resource to remove
        """
        return await self.table.delete(where=f'resource_uri = "{resource_uri}"')

    async def get_all_resource_uris(self) -> list[str]:
        """Retrieve all resource URIs currently stored in the vector database.
        
        Returns:
            List of all resource URIs in the vector store
        """
        results = await self.table.query().select(["resource_uri"]).to_list()
        return [item["resource_uri"] for item in results]

    async def batch_delete_by_uris(self, resource_uris: list[str]):
        """Batch delete multiple resources by their URIs.
        
        Args:
            resource_uris: List of resource URIs to delete from the vector store
        """
        if not resource_uris:
            return
        
        # Build WHERE clause for batch deletion
        uri_conditions = " OR ".join([f'resource_uri = "{uri}"' for uri in resource_uris])
        await self.table.delete(where=uri_conditions)

    async def cleanup_invalid_resources(self, valid_resource_uris: set[str]) -> int:
        """Remove all resources from vector store that are not in the valid set.
        
        Args:
            valid_resource_uris: Set of URIs that should remain in the vector store
            
        Returns:
            Number of invalid resources that were removed
        """
        all_stored_uris = await self.get_all_resource_uris()
        invalid_uris = [uri for uri in all_stored_uris if uri not in valid_resource_uris]
        
        if invalid_uris:
            await self.batch_delete_by_uris(invalid_uris)
            logger.info(f"Deleted {len(invalid_uris)} invalid resources from vector store: " + ", ".join(invalid_uris))
        
        return len(invalid_uris)

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
        vector = await self.embedder.aembed_query(serialized_data)

        return EmbeddingSearch(
            vector=vector,
            resource_uri=element.uri,
            aspect=element.aspect,
            hash=hash_str
        )

    async def _upsert_element(self, resource_uri: str, aspect: str, data: dict,
                              unchecked: bool = False):
        """Internal method to handle element insertion/update."""
        serialized_data = json.dumps(data, ensure_ascii=False, sort_keys=True)
        hash_str = self.hasher.hash(serialized_data)
        existing = None

        if not unchecked:
            existing = await self.get(resource_uri)
            if existing and existing.hash == hash_str:
                return None

        vectors = await self.embedder.aembed_documents([serialized_data])
        vector = vectors[0]

        if existing:
            return await self._update_record(resource_uri, vector, hash_str)

        return await self.table.add([EmbeddingSearch(
            vector=vector,
            resource_uri=resource_uri,
            aspect=aspect,
            hash=hash_str
        )])

    async def _update_element(self, resource_uri: str, aspect: str, data: dict):
        """Internal method to handle element updates."""
        serialized_data = json.dumps(data, ensure_ascii=False, sort_keys=True)
        hash_str = self.hasher.hash(serialized_data)
        vectors = await self.embedder.aembed_documents([serialized_data])

        return await self._update_record(
            resource_uri,
            vectors[0],
            hash_str,
            aspect
        )

    async def _update_record(self, resource_uri: str, vector: list[float],
                             hash_str: str, aspect: Optional[str] = None):
        """Execute update operation on the database table."""
        updates = {'vector': vector, 'hash': hash_str}
        if aspect:
            updates['aspect'] = aspect

        return await self.table.update(
            updates=updates,
            where=f'resource_uri = "{resource_uri}"'
        )
