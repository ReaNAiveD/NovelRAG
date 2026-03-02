import os
from typing import Any, Generator

from langchain_core.embeddings import Embeddings
import logging
import yaml

from novelrag.config.resource import AspectConfig, VectorStoreConfig
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import Element
from novelrag.resource.operation import Operation, PropertyOperation, ResourceOperation
from novelrag.resource.repository import ResourceRepository, SearchResult
from novelrag.storage.local.vector import LanceDBStore

logger = logging.getLogger(__name__)


def _load_element(
    element_data: dict,
    parent_uri: str,
    aspect: str,
    children_keys: list[str],
) -> tuple[Element, list[Element]]:
    """Recursively load an Element and its children from a nested dictionary.

    Uses ``Element.build`` so that *uri*, *aspect* and *children_keys* are
    set correctly and children fields are normalised to ``list[str]``.
    """
    if 'id' not in element_data:
        raise ValueError(
            f"Element data under '{parent_uri}' is missing required 'id' field. "
            f"Got keys: {list(element_data.keys())}. "
            f"Every resource (including items in children_keys lists) must have an 'id' field."
        )
    descendants: list[Element] = []
    # Capture nested children *before* Element.build normalises them to list[str]
    for key in children_keys:
        raw_children = element_data.get(key, [])
        if isinstance(raw_children, list):
            for child_data in raw_children:
                if isinstance(child_data, dict) and 'id' in child_data:
                    child, child_descendants = _load_element(
                        child_data,
                        parent_uri=f'{parent_uri}/{element_data["id"]}',
                        aspect=aspect,
                        children_keys=children_keys,
                    )
                    descendants.append(child)
                    descendants.extend(child_descendants)
    element = Element.build(element_data, parent_uri, aspect, children_keys)
    return element, descendants


class ElementLookUpTable:
    def __init__(self, elements: list[Element]):
        self.table: dict[str, Element] = dict((ele.uri, ele) for ele in elements)

    def find_by_uri(self, uri: str):
        return self.table.get(uri)

    def pop(self, uri: str) -> list[Element] | None:
        element = self.table.pop(uri, None)
        if not element:
            logger.warning(f"Attempted to pop element with URI '{uri}' from lookup table, but it was not found.")
            return None
        popped_elements = [element]
        for child in self.iter_children(element):
            self.table.pop(child.uri, None)
            popped_elements.append(child)
        return popped_elements

    def __getitem__(self, key: str) -> Element:
        return self.table[key]

    def __setitem__(self, key: str, value: Element):
        self.table[key] = value

    def batch_add(self, elements: list[Element]):
        for element in elements:
            self.table[element.uri] = element

    def iter_children(self, element: Element) -> Generator[Element, None, None]:
        """Recursively yield all children of an element."""
        for key in element.children_keys:
            for child_name in element.children_names_of(key):
                child_uri = f'{element.uri}/{child_name}'
                if child_uri not in self.table:
                    logger.warning(
                        f"Element with URI '{child_uri}' not found in lookup table. "
                        f"Referenced as child '{child_name}' under key '{key}' of element '{element.uri}'."
                    )
                    continue
                child = self.table[child_uri]
                yield child
                yield from self.iter_children(child)

    def dump_element(self, element: Element) -> dict:
        """Dump an Element to a nested dictionary format suitable for YAML serialization."""
        data = element.dumped_dict()
        for key in element.children_keys:
            child_names = element.children_names_of(key)
            nested_children = []
            for child_name in child_names:
                child_uri = f'{element.uri}/{child_name}'
                if child_uri not in self.table:
                    logger.warning(
                        f"Element with URI '{child_uri}' not found in lookup table. "
                        f"Referenced as child '{child_name}' under key '{key}' of element '{element.uri}'."
                    )
                    continue
                nested_children.append(self.dump_element(self.table[child_uri]))
            data[key] = nested_children
        return data


class LanceDBResourceRepository(ResourceRepository):
    """Local file-backed repository using LanceDB for vector search.

    * ``ResourceAspect.root_elements`` — ordered list of root element names.
    * ``Element.children_names`` — ordered child-name lists per
      ``children_key``.

    Persistence is to per-aspect YAML files (nested format for
    readability).  The load/save helpers flatten/unflatten accordingly.
    """

    def __init__(
            self,
            aspects_config_path: str,
            aspect_configs: dict[str, AspectConfig],
            aspects: dict[str, ResourceAspect],
            elements: list[Element],
            vector_store: LanceDBStore,
            embedder: Embeddings,
            default_resource_dir: str = '.',
    ):
        # Note: aspects_configs should keep the same keys as resource_aspects, and contain the path info for dumping.
        self.aspects_config_path = aspects_config_path
        self.aspect_configs = aspect_configs
        self.resource_aspects: dict[str, ResourceAspect] = aspects
        self.lut = ElementLookUpTable(elements)
        self.vector_store = vector_store
        self.embedding_llm = embedder
        self.default_resource_dir = default_resource_dir

    @classmethod
    async def load_from_disk(cls,
            aspects_config_path: str,
            vector_store_config: VectorStoreConfig,
            embedder: Embeddings,
            default_resource_dir: str = '.',):
        """Load the repository from disk, populating the aspects and vector store."""
        with open(aspects_config_path, 'r', encoding='utf-8') as f:
            aspects_data = yaml.safe_load(f)
        aspect_configs = {name: AspectConfig.model_validate(data) for name, data in aspects_data.items()}
        aspects = {}
        all_elements = []
        for name, aspect_config in aspect_configs.items():
            aspect, elements = await cls._load_aspect_from_disk(name, aspect_config)
            aspects[name] = aspect
            all_elements.extend(elements)
        # Create vector store
        vector_store = await LanceDBStore.create(
            uri=vector_store_config.lancedb_uri,
            table_name=vector_store_config.table_name,
            embedder=embedder,
        )
        if vector_store_config.cleanup_invalid_on_init:
            invalid_count = await vector_store.cleanup_invalid_resources(valid_uris=set(ele.uri for ele in all_elements))
            if invalid_count > 0:
                logger.info(f"Cleaned up {invalid_count} invalid vectors from the vector store.")
        await vector_store.batch_add(all_elements)
        return cls(
            aspects_config_path=aspects_config_path,
            aspect_configs=aspect_configs,
            aspects=aspects,
            elements=all_elements,
            vector_store=vector_store,
            embedder=embedder,
            default_resource_dir=default_resource_dir,
        )

    @staticmethod
    async def _load_aspect_from_disk(name: str, aspect: AspectConfig) -> tuple[ResourceAspect, list[Element]]:
        """Load root elements for a given aspect from disk."""
        root_elements = []
        all_elements = []
        if not os.path.exists(aspect.path):
            logger.warning(f"Aspect file for aspect '{name}' not found at path '{aspect.path}'. Initializing with empty root elements.")
            aspect_obj = ResourceAspect.from_config(name, aspect, root_elements=[])
            return aspect_obj, []
        with open(aspect.path, 'r', encoding='utf-8') as f:
            elements_data = yaml.safe_load(f)
        for root_element_data in elements_data:
            root_element, children = _load_element(root_element_data, parent_uri=f'/{name}', aspect=name, children_keys=aspect.children_keys)
            root_elements.append(root_element.id)
            all_elements.append(root_element)
            all_elements.extend(children)
        aspect_obj = ResourceAspect.from_config(name, aspect, root_elements=root_elements)
        return aspect_obj, all_elements

    async def _dump_aspects(self):
        """Dump all aspects to the registry."""
        aspect_configs_data = {name: config.model_dump() for name, config in self.aspect_configs.items()}
        with open(self.aspects_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(aspect_configs_data, f)

    async def _dump_aspect_elements(self, config: AspectConfig, aspect: ResourceAspect):
        """Dump the elements of a given aspect to disk."""
        elements_data = []
        for root_element_name in aspect.root_elements:
            root_uri = f'/{aspect.name}/{root_element_name}'
            if root_uri not in self.lut.table:
                logger.warning(
                    f"Root element with URI '{root_uri}' not found in lookup table. "
                    f"Referenced as root element '{root_element_name}' of aspect '{aspect.name}'."
                )
                continue
            root_element = self.lut[root_uri]
            elements_data.append(self.lut.dump_element(root_element))
        with open(config.path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(elements_data, f)

    async def _dump_with_uri(self, uri: str):
        """Dump the aspect file corresponding to the given URI."""
        if not uri.startswith('/'):
            logger.warning(f"Invalid URI '{uri}' for dumping. Must start with '/'.")
            return
        if uri == '/':
            await self._dump_aspects()
            for aspect_name in self.aspect_configs.keys():
                aspect = self.resource_aspects.get(aspect_name)
                config = self.aspect_configs.get(aspect_name)
                if aspect and config:
                    await self._dump_aspect_elements(config, aspect)
                else:
                    logger.warning(f"Aspect '{aspect_name}' or its config not found for dumping.")
        elif uri[1:] in self.resource_aspects:
            await self._dump_aspects()
            aspect = self.resource_aspects.get(uri[1:])
            config = self.aspect_configs.get(uri[1:])
            if aspect and config:
                await self._dump_aspect_elements(config, aspect)
            else:
                logger.warning(f"Aspect '{uri[1:]}' or its config not found for dumping.")
        else:
            aspect_name = uri.split('/')[1]
            aspect = self.resource_aspects.get(aspect_name)
            config = self.aspect_configs.get(aspect_name)
            if aspect and config:
                await self._dump_aspect_elements(config, aspect)
            else:
                logger.warning(f"Aspect '{aspect_name}' or its config not found for dumping.")

    def _find_valid_path(self, aspect_name: str) -> str:
        base_path = os.path.join(self.default_resource_dir, f"{aspect_name}.yml")
        if not os.path.exists(base_path):
            return base_path
        index = 0
        while True:
            indexed_path = os.path.join(self.default_resource_dir, f"{aspect_name}_{index}.yml")
            if not os.path.exists(indexed_path):
                return indexed_path
            index += 1

    def _iter_elements_of_aspect(self, aspect: ResourceAspect) -> Generator[Element, None, None]:
        for root_element_name in aspect.root_elements:
            root_uri = f'/{aspect.name}/{root_element_name}'
            if root_uri not in self.lut.table:
                logger.warning(
                    f"Root element with URI '{root_uri}' not found in lookup table. "
                    f"Referenced as root element '{root_element_name}' of aspect '{aspect.name}'."
                )
                continue
            root_element = self.lut[root_uri]
            yield root_element
            yield from self.lut.iter_children(root_element)

    async def all_aspects(self) -> list[ResourceAspect]:
        """Return a list of all ResourceAspects in the repository."""
        return list(self.resource_aspects.values())

    async def get_aspect(self, name: str) -> ResourceAspect | None:
        """Get a ResourceAspect by name, or None if it doesn't exist."""
        return self.resource_aspects.get(name)

    async def add_aspect(self, name: str, metadata: dict[str, Any]) -> ResourceAspect:
        """Add a new aspect to the repository."""
        if name in self.resource_aspects:
            raise ValueError(f"Aspect with name '{name}' already exists.")
        if 'path' not in metadata:
            metadata['path'] = self._find_valid_path(name)
        config = AspectConfig.model_validate(metadata)
        aspect = ResourceAspect.from_config(name, config)
        self.aspect_configs[name] = config
        self.resource_aspects[name] = aspect
        await self._dump_aspects()
        return aspect

    async def remove_aspect(self, name: str) -> ResourceAspect | None:
        """Remove an aspect and its elements from the repository."""
        aspect = self.resource_aspects.pop(name, None)
        if aspect:
            popped_elements = []
            for root_element_name in aspect.root_elements:
                root_uri = f'/{aspect.name}/{root_element_name}'
                popped = self.lut.pop(root_uri)
                if popped:
                    popped_elements.extend(popped)
            if popped_elements:
                await self.vector_store.batch_delete_by_uris([e.uri for e in popped_elements])
            config = self.aspect_configs.pop(name, None)
            if config and os.path.exists(config.path):
                os.remove(config.path)
            await self._dump_aspects()
        return aspect

    async def iter_elements(self, aspect_name: str) -> list[Element]:
        aspect = self.resource_aspects.get(aspect_name)
        if not aspect:
            return []
        return list(self._iter_elements_of_aspect(aspect))

    async def find_by_uri(self, resource_uri: str) -> list[str] | ResourceAspect | Element | None:
        if resource_uri and resource_uri == '/':
            return list(self.resource_aspects.keys())
        elif resource_uri and resource_uri.startswith('/') and resource_uri[1:] in self.resource_aspects:
            return self.resource_aspects[resource_uri[1:]]
        return self.lut.find_by_uri(resource_uri)

    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = 20) -> list[SearchResult]:
        vector = await self.embedding_llm.aembed_query(query)
        result = await self.vector_store.vector_search(vector, aspect=aspect, limit=limit)
        return [SearchResult(distance=item.distance, element=self.lut[item.resource_uri]) for item in result]

    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository.
        
        Args:
            op: The operation to apply
        """
        if isinstance(op, PropertyOperation):
            resource = await self.find_by_uri(op.resource_uri)
            if isinstance(resource, Element):
                undo = resource.update(op.data)
                await self.vector_store.update(resource)
                await self._dump_with_uri(resource.uri)
                return op.create_undo(undo)
            elif isinstance(resource, ResourceAspect):
                undo = resource.update(op.data)
                # Avoid dumping the entire element tree
                await self._dump_aspects()
                return op.create_undo(undo)
            elif isinstance(resource, list):
                raise ValueError(f"Cannot apply PropertyOperation to aspect list at URI '{op.resource_uri}'.")
            else:
                raise ValueError(f"Resource at URI '{op.resource_uri}' not found for PropertyOperation.")
        elif isinstance(op, ResourceOperation):
            target = await self.find_by_uri(op.location.resource_uri)
            new_elements = []
            new_children = []
            aspect_name = op.location.resource_uri.strip('/').split('/')[0]
            aspect_obj = self.resource_aspects.get(aspect_name)
            ck = aspect_obj.children_keys if aspect_obj else []
            for element_data in op.data or []:
                element, children = _load_element(element_data, parent_uri=op.location.resource_uri, aspect=aspect_name, children_keys=ck)
                new_elements.append(element)
                new_children.extend(children)
            new_children_names = [e.id for e in new_elements]
            if isinstance(target, Element):
                if op.location.children_key is None:
                    raise ValueError(f"ResourceOperation with target 'resource' must specify a children_key for element targets.")
                current_children = target.children_names_of(op.location.children_key)
                undo_uris = [f'{target.uri}/{name}' for name in current_children[op.start:op.end]]
                # Splice the children list
                target.set_children_names(op.location.children_key, current_children[:op.start] + new_children_names + current_children[op.end:])
            elif isinstance(target, ResourceAspect):
                if op.location.children_key is not None:
                    logger.warning(f"ResourceOperation with target 'resource' on aspect '{target.name}' should not specify a children_key. Ignoring children_key '{op.location.children_key}'.")
                current_children = target.root_elements
                undo_uris = [f'/{target.name}/{name}' for name in current_children[op.start:op.end]]
                # Splice the root elements list
                target.root_elements = current_children[:op.start] + new_children_names + current_children[op.end:]
            elif isinstance(target, list):
                raise ValueError(f"Cannot apply ResourceOperation to aspect list at URI '{op.location.resource_uri}'.")
            else:
                raise ValueError(f"Resource at URI '{op.location.resource_uri}' not found for ResourceOperation.")
            undo_data = []
            popped_elements = []
            for child_uri in undo_uris:
                if child_uri not in self.lut.table:
                    logger.warning(
                        f"Element with URI '{child_uri}' not found in lookup table."
                    )
                    continue
                undo_data.append(self.lut.dump_element(self.lut[child_uri]))
                popped_elements.extend(self.lut.pop(child_uri) or [])
            if popped_elements:
                await self.vector_store.batch_delete_by_uris([e.uri for e in popped_elements])
            await self.vector_store.batch_add(new_elements + new_children)
            self.lut.batch_add(new_elements + new_children)
            await self._dump_with_uri(op.location.resource_uri)
            return op.create_undo(undo_data)
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")

    async def update_relationships(self, source_uri: str, target_uri: str, relationships: list[str]) -> list[str]:
        """Update the relationships of a resource by its URI.

        Args:
            source_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relationships: A list of relations to set for the resource

        Returns:
            List[str]: The old relationships before the update
        """
        resource = await self.find_by_uri(source_uri)
        if not isinstance(resource, Element):
            raise ValueError(f"Resource at URI '{source_uri}' is not an Element.")
        old_relationships = resource.update_relationships(target_uri, relationships)
        await self._dump_with_uri(source_uri)
        return old_relationships
