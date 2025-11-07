import os.path
from typing import Generator, Any

import yaml

from novelrag.config.resource import AspectConfig

from .element import DirectiveElement, Element, DirectiveElementList


class ResourceAspect:
    def  __init__(self, name: str, path: str, children_keys: list[str], description: str | None = None, metadata: dict[str, Any] | None = None):
        self.name = name
        self.description = description
        self.path = path
        self.children_keys = children_keys
        self.metadata = metadata or {}
        self.root_elements: DirectiveElementList = DirectiveElementList()

    @classmethod
    def from_config(cls, name: str, config: AspectConfig) -> 'ResourceAspect':
        """Create a ResourceAspect from an AspectConfig object."""
        return cls(
            name=name,
            description=config.description,
            path=config.path,
            children_keys=config.children_keys,
            metadata=config.model_extra,
        )
    
    def to_config(self) -> AspectConfig:
        """Convert the ResourceAspect to an AspectConfig object."""
        return AspectConfig.model_validate({
            "path": self.path,
            "children_keys": self.children_keys,
            **({"description": self.description} if self.description else {}),
            **self.metadata,  # Include any additional metadata as extra fields
        })

    def _load_raw_content(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _dump_content(self):
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            yaml.safe_dump([ele.inner.dumped_dict() for ele in self.root_elements], f, allow_unicode=True)

    def load_from_file(self):
        raw_content = self._load_raw_content() or []
        assert isinstance(raw_content, list)
        elements = [Element.build(ele, f'/{self.name}', self.name, self.children_keys) for ele in raw_content]
        self.root_elements = DirectiveElementList.wrap(elements, self.children_keys)

    def save_to_file(self):
        if len(self.root_elements) > 0 or os.path.exists(self.path):
            self._dump_content()

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
    
    @property
    def aspect_dict(self):
        """Returns a dictionary representation of the aspect."""
        return {
            'name': self.name,
            **({'description': self.description} if self.description else {}),
            **self.metadata,  # Include any additional metadata as extra fields
        }

    @property
    def context_dict(self):
        """Returns a dictionary composed of name, path, and children_keys"""
        return {
            'name': self.name,
            'children_keys': self.children_keys,
            **({"description": self.description} if self.description else {}),
            **self.metadata,  # Include any additional metadata as extra fields
            'root_elements': [{"id": ele.id} for ele in self.root_elements],
        }
