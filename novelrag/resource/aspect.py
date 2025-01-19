import os.path
from typing import Generator, Any

import yaml

from novelrag.llm.oai.embedding import OpenAIEmbeddingLLM
from novelrag.resource.element import DirectiveElement, Element, DirectiveElementList


class ResourceAspect:
    def  __init__(self, name: str, path: str, children_keys: list[str]):
        self.aspect_name = name
        self.path = path
        self.children_keys = children_keys
        self.root_elements: DirectiveElementList = DirectiveElementList()

    def _load_raw_content(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_from_file(self):
        raw_content = self._load_raw_content() or []
        assert isinstance(raw_content, list)
        elements = [Element.build(ele, self.aspect_name, self.children_keys) for ele in raw_content]
        self.root_elements = DirectiveElementList.wrap(elements, self.children_keys)

    async def ensure_embeddings(self, embedder: OpenAIEmbeddingLLM):
        return await self.root_elements.ensure_embeddings(embedder)

    def save_to_file(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            yaml.safe_dump([ele.inner.model_dump() for ele in self.root_elements], f, allow_unicode=True)

    def iter_elements(self) -> Generator[DirectiveElement, Any, None]:
        """Iterate through all elements in the aspect tree in depth-first order.
        
        Yields:
            DirectiveElement: Each element in the tree, starting from root elements
                            and traversing through their children recursively.
        """
        def _iter_element_tree(element: DirectiveElement):
            yield element
            for elements in element.children.values():
                for element in elements:
                    yield from _iter_element_tree(element)
        
        for root_element in self.root_elements:
            yield from _iter_element_tree(root_element)

    def splice(self, start: int, end: int, *items: 'Element') -> tuple[list['DirectiveElement'], list['DirectiveElement']]:
        old = self.root_elements[start: end]
        wrapped = DirectiveElementList.wrap(elements=list(items), children_keys=self.children_keys)
        self.root_elements = self.root_elements.splice(start, end, *wrapped)
        return wrapped, old
