from dataclasses import dataclass

from novelrag.config.resource import ResourceConfig, VectorStoreConfig
from novelrag.exceptions import ElementNotFoundError, OperationError
from novelrag.llm.oai.embedding import OpenAIEmbeddingLLM
from novelrag.resource import DirectiveElement, Element
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.lut import ElementLookUpTable
from novelrag.resource.operation import Operation, ElementOperation, \
    PropertyOperation, ElementLocation, AspectLocation
from novelrag.resource.vector import LanceDBStore, EmbeddingSearch


@dataclass
class SearchResult:
    distance: float
    element: DirectiveElement


class ResourceRepository:
    def __init__(
            self,
            resource_aspects: dict[str, ResourceAspect],
            vector_store: LanceDBStore,
            embedder: OpenAIEmbeddingLLM,
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
            embedding_config: dict,
    ) -> 'ResourceRepository':
        """Build a ResourceRepository from a dictionary of aspect configurations.
        
        Args:
            resource_config: Dictionary mapping aspect names to their ResourceConfig objects
            vector_store_config: configuration of vector store
            embedding_config: configuration of embedding llm
            
        Returns:
            A new ResourceRepository instance initialized with the configured aspects
        """
        embedder = OpenAIEmbeddingLLM.from_config(
            config=embedding_config,
        )
        aspects = {}
        vector_items = []
        for aspect_name, aspect_config in resource_config.items():
            aspect = ResourceAspect(
                name=aspect_name,
                path=aspect_config.path,
                children_keys=aspect_config.children_keys
            )
            aspect.load_from_file()
            aspects[aspect_name] = aspect
            await aspect.ensure_embeddings(embedder)
            for element in aspect.iter_elements():
                vector_items.append(EmbeddingSearch(
                    vector=element.embedding,
                    element_id=element.id,
                    aspect=element.inner.aspect,
                ))
        vector_store = await LanceDBStore.create(
            vector_store_config.lancedb_uri,
            vector_store_config.table_name,
            init_data=vector_items,
            overwrite=vector_store_config.overwrite,
        )

        return cls(aspects, vector_store, embedder)

    def _rebuild_lut(self):
        elements = [
            element
            for aspect in self.resource_aspects.values()
            for element in aspect.iter_elements()
        ]
        self.lut = ElementLookUpTable(elements)

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = 20):
        embeddings = await self.embedding_llm.embedding(query)
        vector = embeddings[0]
        result = await self.vector_store.vector_search(vector, aspect=aspect, limit=limit)
        return [SearchResult(distance=item['_distance'], element=self.lut[item['element_id']]) for item in result]

    async def apply(self, op: Operation):
        if isinstance(op, ElementOperation):
            if isinstance(op.location, ElementLocation):
                ele = self.lut.find_by_id(op.location.element_id)
                if not ele:
                    raise ElementNotFoundError(op.element_id)
                data = [Element.build(item, ele.inner.aspect, ele.inner.children_keys) for item in (op.data or [])]
                new, old = ele.splice_at(op.location.children_key, op.start, op.end, *data)
            elif isinstance(op.location, AspectLocation):
                aspect = self.resource_aspects[op.location.aspect]
                data = [Element.build(item, op.location.aspect, aspect.children_keys) for item in (op.data or [])]
                new, old = aspect.splice(op.start, op.end, *data)
            else:
                raise OperationError(f'Unrecognized Operation Location Type: {type(op.location)}')
            for ele in old:
                await self._handle_deleted(ele)
            for ele in new:
                await self._handle_added(ele)
        elif isinstance(op, PropertyOperation):
            ele = self.lut.find_by_id(op.element_id)
            if not ele:
                raise ElementNotFoundError(op.element_id)
            undo = ele.update(op.data)
            await self._handle_updated(ele)
            return op.create_undo(undo)

    async def _handle_added(self, ele: DirectiveElement):
        self.lut[ele.id] = ele
        await ele.ensure_embedding(self.embedding_llm)
        await self.vector_store.add(ele.id, ele.embedding, ele.inner.aspect)

    async def _handle_deleted(self, ele: DirectiveElement):
        self.lut.pop(ele.id)
        await self.vector_store.delete(ele_id=ele.id)
        for children in ele.children.values():
            for child in children:
                await self._handle_deleted(child)

    async def _handle_updated(self, ele: DirectiveElement):
        await ele.update_embedding(self.embedding_llm)
        await self.vector_store.update_vector(ele.id, ele.embedding)
