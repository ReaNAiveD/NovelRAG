from abc import ABC, abstractmethod
from dataclasses import dataclass

from novelrag.config.resource import ResourceConfig, VectorStoreConfig
from novelrag.exceptions import ElementNotFoundError, OperationError
from novelrag.llm import EmbeddingLLM
from .aspect import ResourceAspect
from .element import DirectiveElement, Element
from .lut import ElementLookUpTable
from .operation import Operation, ElementOperation, \
    PropertyOperation, ElementLocation, AspectLocation
from .vector import LanceDBStore


@dataclass
class SearchResult:
    distance: float
    element: DirectiveElement


class ResourceRepository(ABC):
    @abstractmethod
    async def get_aspect(self, name: str) -> ResourceAspect | None:
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
    async def find_by_uri(self, element_uri: str) -> DirectiveElement | None:
        """Find an element by its ID in the repository.
        
        Args:
            element_uri: The ID of the element to find
        
        Returns:
            The DirectiveElement if found, otherwise None
        """
        pass

    @abstractmethod
    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository.
        
        Args:
            op: The operation to apply
        """
        pass


class LanceDBResourceRepository(ResourceRepository):
    def __init__(
            self,
            resource_aspects: dict[str, ResourceAspect],
            vector_store: LanceDBStore,
            embedder: EmbeddingLLM,
    ):
        self.resource_aspects: dict[str, ResourceAspect] = resource_aspects
        elements = [
            element 
            for aspect in self.resource_aspects.values()
            for element in aspect.iter_elements()
        ]
        self.lut = ElementLookUpTable(elements)
        self.vector_store = vector_store
        self.embedding_llm = embedder

    @classmethod
    async def from_config(
            cls,
            resource_config: dict[str, ResourceConfig],
            vector_store_config: VectorStoreConfig,
            embedder: EmbeddingLLM,
    ) -> 'LanceDBResourceRepository':
        """Build a ResourceRepository from a dictionary of aspect configurations.
        
        Args:
            resource_config: Dictionary mapping aspect names to their ResourceConfig objects
            vector_store_config: configuration of vector store
            embedder: embedding llm
            
        Returns:
            A new ResourceRepository instance initialized with the configured aspects
        """
        aspects = {}
        elements = []
        for aspect_name, aspect_config in resource_config.items():
            aspect = ResourceAspect(
                name=aspect_name,
                path=aspect_config.path,
                children_keys=aspect_config.children_keys
            )
            aspect.load_from_file()
            aspects[aspect_name] = aspect
            for element in aspect.iter_elements():
                elements.append(element.inner)
        vector_store = await LanceDBStore.create(
            uri=vector_store_config.lancedb_uri,
            table_name=vector_store_config.table_name,
            embedder=embedder,
        )
        await vector_store.batch_add(elements)

        return cls(aspects, vector_store, embedder)

    def _rebuild_lut(self):
        elements = [
            element
            for aspect in self.resource_aspects.values()
            for element in aspect.iter_elements()
        ]
        self.lut = ElementLookUpTable(elements)

    async def get_aspect(self, name: str) -> ResourceAspect | None:
        return self.resource_aspects.get(name)
    
    async def find_by_uri(self, element_uri: str) -> DirectiveElement | None:
        """Find an element by its ID in the repository.
        
        Args:
            element_uri: The ID of the element to find
        
        Returns:
            The DirectiveElement if found, otherwise None
        """
        return self.lut.find_by_uri(element_uri)

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = 20) -> list[SearchResult]:
        embeddings = await self.embedding_llm.embedding(query)
        vector = embeddings[0]
        result = await self.vector_store.vector_search(vector, aspect=aspect, limit=limit)
        return [SearchResult(distance=item.distance, element=self.lut[item.element_uri]) for item in result]

    async def apply(self, op: Operation) -> Operation:
        if isinstance(op, ElementOperation):
            if isinstance(op.location, ElementLocation):
                ele = self.lut.find_by_uri(op.location.element_uri)
                if not ele:
                    raise ElementNotFoundError(op.location.element_uri)
                data = [Element.build(item, ele.uri, ele.aspect, ele.inner.children_keys) for item in (op.data or [])]
                new, old = ele.splice_at(op.location.children_key, op.start, op.end, *data)
            elif isinstance(op.location, AspectLocation):
                aspect = self.resource_aspects[op.location.aspect]
                data = [Element.build(item, op.location.aspect, op.location.aspect, aspect.children_keys) for item in (op.data or [])]
                new, old = aspect.splice(op.start, op.end, *data)
            else:
                raise OperationError(f'Unrecognized Operation Location Type: {type(op.location)}')
            for ele in old:
                await self._handle_deleted(ele)
            for ele in new:
                await self._handle_added(ele)
            return op.create_undo([item.inner.dumped_dict() for item in old])
        elif isinstance(op, PropertyOperation):
            ele = self.lut.find_by_uri(op.element_uri)
            if not ele:
                raise ElementNotFoundError(op.element_uri)
            undo = ele.update(op.data)
            await self._handle_updated(ele)
            return op.create_undo(undo)

    async def _handle_added(self, ele: DirectiveElement):
        self.lut[ele.uri] = ele
        await self.vector_store.add(ele.inner)
        self.resource_aspects[ele.inner.aspect].save_to_file()

    async def _handle_deleted(self, ele: DirectiveElement):
        self.lut.pop(ele.uri)
        await self.vector_store.delete(element_uri=ele.uri)
        for children in ele.children.values():
            for child in children:
                await self._handle_deleted(child)
        self.resource_aspects[ele.inner.aspect].save_to_file()

    async def _handle_updated(self, ele: DirectiveElement):
        await self.vector_store.update(ele.inner)
        self.resource_aspects[ele.inner.aspect].save_to_file()
