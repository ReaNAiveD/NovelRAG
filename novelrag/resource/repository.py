from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any

import yaml

from novelrag.config.resource import AspectConfig, VectorStoreConfig
from novelrag.exceptions import ElementNotFoundError, OperationError
from novelrag.llm import EmbeddingLLM
from .aspect import ResourceAspect
from .element import DirectiveElement, Element
from .lut import ElementLookUpTable
from .operation import Operation, ResourceOperation, \
    PropertyOperation
from .vector import LanceDBStore


@dataclass
class SearchResult:
    distance: float
    element: DirectiveElement


class ResourceRepository(ABC):
    @abstractmethod
    async def all_aspects(self) -> list[ResourceAspect]:
        pass

    @abstractmethod
    async def get_aspect(self, name: str) -> ResourceAspect | None:
        pass

    @abstractmethod
    def add_aspect(self, name: str, metadata: dict[str, Any]) -> ResourceAspect:
        pass

    @abstractmethod
    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = None) -> list[SearchResult]:
        """Search for resources using vector similarity.
        
        Args:
            query: The search query string
            aspect: Optional aspect to filter results
            limit: Maximum number of results to return
        """
        pass

    @abstractmethod
    async def find_by_uri(self, resource_uri: str) -> list[ResourceAspect] | ResourceAspect | DirectiveElement | None:
        """Find a resource by its URI in the repository.
        
        Args:
            resource_uri: The URI of the resource to find
        
        Returns:
            - list[ResourceAspect]: All aspects if resource_uri is '/'
            - ResourceAspect: Single aspect if resource_uri matches '/{aspect_name}'
            - DirectiveElement: Element if found by URI
            - None: If no match is found
        """
        pass

    @abstractmethod
    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository.
        
        Args:
            op: The operation to apply
        """
        pass

    @abstractmethod
    async def update_relations(self, resource_uri: str, target_uri: str, relations: list[str]) -> Element:
        """Update the relations of a resource by its URI.
        Args:
            resource_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relations: A dictionary of relations to set for the resource
        Returns:
            Element: The updated element with new relations
        """
        pass


class LanceDBResourceRepository(ResourceRepository):
    def __init__(
            self,
            config_path: str,
            resource_aspects: dict[str, ResourceAspect],
            vector_store: LanceDBStore,
            embedder: EmbeddingLLM,
            default_resource_dir: str = '.',
    ):
        self.config_path = config_path
        self.resource_aspects: dict[str, ResourceAspect] = resource_aspects
        elements = [
            element 
            for aspect in self.resource_aspects.values()
            for element in aspect.iter_elements()
        ]
        self.lut = ElementLookUpTable(elements)
        self.vector_store = vector_store
        self.embedding_llm = embedder
        self.default_resource_dir = default_resource_dir

    @classmethod
    async def from_config(
            cls,
            config_path: str,
            vector_store_config: VectorStoreConfig,
            embedder: EmbeddingLLM,
            default_resource_dir: str = '.',
            cleanup_invalid_vectors: bool | None = None,
    ) -> 'LanceDBResourceRepository':
        """Build a LanceDBResourceRepository from a YAML configuration file.

        This method loads resource aspects from a YAML configuration file, creates
        a vector store, and populates it with all elements from the loaded aspects.
        It also cleans up any invalid vectors that no longer exist in the configuration.

        Args:
            config_path: Path to the YAML configuration file containing aspect definitions
            vector_store_config: Configuration object for the LanceDB vector store
            embedder: Embedding LLM instance used for generating vector embeddings
            default_resource_dir: Default directory for resource files (defaults to '.')
            cleanup_invalid_vectors: Whether to remove invalid vectors during initialization.
                                   If None, uses vector_store_config.cleanup_invalid_on_init

        Returns:
            A new LanceDBResourceRepository instance initialized with the configured
            aspects and a populated vector store

        Raises:
            FileNotFoundError: If the config_path file doesn't exist
            yaml.YAMLError: If the configuration file contains invalid YAML
            ValidationError: If aspect configurations don't match the expected schema
        """
        aspects = {}
        elements = []
        
        # Load aspects and collect valid elements
        with open(config_path, 'r', encoding='utf-8') as f:
            resource_config: dict = yaml.safe_load(f) or {}
        
        for aspect_name, aspect_config in resource_config.items():
            aspect_config = AspectConfig.model_validate(aspect_config)
            aspect = ResourceAspect.from_config(aspect_name, aspect_config)
            aspect.load_from_file()
            aspects[aspect_name] = aspect
            for element in aspect.iter_elements():
                elements.append(element.inner)
        
        # Create vector store
        vector_store = await LanceDBStore.create(
            uri=vector_store_config.lancedb_uri,
            table_name=vector_store_config.table_name,
            embedder=embedder,
        )
        
        # Determine if cleanup should be performed
        should_cleanup = cleanup_invalid_vectors if cleanup_invalid_vectors is not None else vector_store_config.cleanup_invalid_on_init

        # Clean up invalid vectors if requested
        if should_cleanup:
            valid_resource_uris = {element.uri for element in elements}
            invalid_count = await vector_store.cleanup_invalid_resources(valid_resource_uris)
            if invalid_count > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Cleaned up {invalid_count} invalid vectors from database")
        
        # Add current elements to vector store
        await vector_store.batch_add(elements)

        return cls(config_path, aspects, vector_store, embedder, default_resource_dir)

    def dump_config(self):
        """Convert the repository to a dictionary of AspectConfig objects."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump({
                name: aspect.to_config().model_dump()
                for name, aspect in self.resource_aspects.items()
            }, f, indent=2, allow_unicode=True, encoding='utf-8')

    def _rebuild_lut(self):
        elements = [
            element
            for aspect in self.resource_aspects.values()
            for element in aspect.iter_elements()
        ]
        self.lut = ElementLookUpTable(elements)

    async def all_aspects(self) -> list[ResourceAspect]:
        return list(self.resource_aspects.values())

    async def get_aspect(self, name: str) -> ResourceAspect | None:
        return self.resource_aspects.get(name)
    
    def add_aspect(self, name: str, metadata: dict[str, Any]) -> ResourceAspect:
        """Add a new aspect to the repository.
        
        This method creates a new ResourceAspect with the given name and metadata,
        automatically assigning a file path if not provided. If a file with the
        default name already exists, it appends an index to avoid conflicts.
        
        Args:
            name: The name of the aspect to add
            metadata: Dictionary containing aspect configuration data.
                     If 'path' is not provided, it will be auto-generated.
        
        Returns:
            ResourceAspect: The newly created and added aspect
            
        Note:
            - The aspect configuration is validated using AspectConfig
            - The repository configuration is automatically saved after adding
            - File paths are generated in the format: {name}.yml or {name}_{index}.yml
        """
        if 'path' not in metadata:
            # Generate a unique file path for the aspect
            base_path = os.path.join(self.default_resource_dir, f"{name}.yml")
            if not os.path.exists(base_path):
                metadata['path'] = base_path
            else:
                # If file exists, append an index to create a unique filename
                index = 0
                while True:
                    indexed_path = os.path.join(self.default_resource_dir, f"{name}_{index}.yml")
                    if not os.path.exists(indexed_path):
                        metadata['path'] = indexed_path
                        break
                    index += 1
        
        # Validate and create the aspect
        aspect_config = AspectConfig.model_validate(metadata)
        aspect = ResourceAspect.from_config(name, aspect_config)

        # Add to repository and save configuration
        self.resource_aspects[name] = aspect
        self.dump_config()

        return aspect
    
    async def find_by_uri(self, resource_uri: str) -> list[ResourceAspect] | ResourceAspect | DirectiveElement | None:
        """Find a resource by its URI in the repository.
        
        Args:
            resource_uri: The URI of the resource to find
        
        Returns:
            - list[ResourceAspect]: All aspects if resource_uri is '/'
            - ResourceAspect: Single aspect if resource_uri matches '/{aspect_name}'
            - DirectiveElement: Element if found by URI
            - None: If no match is found
        """
        if resource_uri and resource_uri == '/':
            return list(self.resource_aspects.values())
        elif resource_uri and resource_uri.startswith('/') and resource_uri[1:] in self.resource_aspects:
            return self.resource_aspects[resource_uri[1:]]
        return self.lut.find_by_uri(resource_uri)

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = 20) -> list[SearchResult]:
        embeddings = await self.embedding_llm.embedding(query)
        vector = embeddings[0]
        result = await self.vector_store.vector_search(vector, aspect=aspect, limit=limit)
        return [SearchResult(distance=item.distance, element=self.lut[item.resource_uri]) for item in result]

    async def apply(self, op: Operation) -> Operation:
        if isinstance(op, ResourceOperation):
            if op.location.children_key is not None:
                # Operating on a resource's children list
                ele = self.lut.find_by_uri(op.location.resource_uri)
                if not ele:
                    raise ElementNotFoundError(op.location.resource_uri)
                data = [Element.build(item, ele.uri, ele.aspect, ele.inner.children_keys) for item in (op.data or [])]
                new, old = ele.splice_at(op.location.children_key, op.start, op.end, *data)
            else:
                # Operating on an aspect's root list
                aspect_name = op.location.resource_uri.strip('/')
                aspect = self.resource_aspects[aspect_name]
                data = [Element.build(item, aspect_name, aspect_name, aspect.children_keys) for item in (op.data or [])]
                new, old = aspect.splice(op.start, op.end, *data)
            for ele in old:
                await self._handle_deleted(ele)
            for ele in new:
                await self._handle_added(ele)
            return op.create_undo([item.inner.dumped_dict() for item in old])
        elif isinstance(op, PropertyOperation):
            ele = self.lut.find_by_uri(op.resource_uri)
            if not ele:
                raise ElementNotFoundError(op.resource_uri)
            undo = ele.update(op.data)
            await self._handle_updated(ele)
            return op.create_undo(undo)

    async def update_relations(self, resource_uri: str, target_uri: str, relations: list[str]) -> Element:
        """Update the relations of a resource by its URI.

        Args:
            resource_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relations: A dictionary of relations to set for the resource

        Returns:
            Element: The updated element with new relations
        """
        ele = self.lut.find_by_uri(resource_uri)
        if not isinstance(ele, DirectiveElement):
            raise ElementNotFoundError(resource_uri)
        _ = ele.update_relationships(target_uri, relations)
        # await self._handle_updated(ele)
        return ele.inner

    async def _handle_added(self, ele: DirectiveElement):
        self.lut[ele.uri] = ele
        await self.vector_store.add(ele.inner)
        self.resource_aspects[ele.inner.aspect].save_to_file()

    async def _handle_deleted(self, ele: DirectiveElement):
        self.lut.pop(ele.uri)
        await self.vector_store.delete(resource_uri=ele.uri)
        for children in ele.children.values():
            for child in children:
                await self._handle_deleted(child)
        self.resource_aspects[ele.inner.aspect].save_to_file()

    async def _handle_updated(self, ele: DirectiveElement):
        await self.vector_store.update(ele.inner)
        self.resource_aspects[ele.inner.aspect].save_to_file()
