import json
import logging

from dataclasses import dataclass
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from novelrag.exceptions import ChildrenKeyNotFoundError

logger = logging.getLogger(__name__)


class Element(BaseModel):
    id: Annotated[str, Field(description='Id of the element')]
    uri: Annotated[str, Field(description='URI of the element')]
    relations: Annotated[dict[str, list[str]], Field(description='Related Elements. <Id>: <Description>', default_factory=lambda: {})]
    aspect: Annotated[str, Field(description='Aspect of the element')]
    children_keys: Annotated[list[str], Field(default_factory=lambda: [])]

    model_config = ConfigDict(extra='allow')

    @classmethod
    def build(
            cls,
            value: dict,
            parent_uri: str,
            aspect: str,
            children_keys: list[str],
            *,
            strict: bool | None = None,
            from_attributes: bool | None = None,
            context: Any | None = None
    ):
        """
        Build an Element instance from a dictionary value.
        Note that the value must contain 'id', 'relations', 'aspect', and 'children_keys' keys.
        """
        uri = f'{parent_uri}/{value["id"]}'
        value['uri'] = uri
        value['aspect'] = aspect
        value['children_keys'] = children_keys
        ele = cls.model_validate(value, strict=strict, from_attributes=from_attributes, context=context)
        for key in children_keys:
            if ele.model_extra and key in ele.model_extra and isinstance(ele.model_extra[key], list):
                ele.model_extra[key] = [cls.build(child, uri, aspect, children_keys, strict=strict, from_attributes=from_attributes, context=context) for child in ele.model_extra[key]]
        return ele

    def props(self):
        """
        Returns a dictionary of properties excluding children keys.
        Usually includes all properties in model_extra except those defined in children_keys.
        Excludes properties like 'id', 'relations', 'aspect', and 'children_keys'.
        """
        return dict((k, v) for k, v in self.model_extra.items() if k not in self.children_keys) if self.model_extra else {}
    
    def flattened_child_ids(self):
        return dict((key, [child.id for child in self.children_of(key)]) for key in self.children_keys)

    def children_ids(self):
        """Returns id of children elements only"""
        return dict((key, [{"id": child.id} for child in self.children_of(key)]) for key in self.children_keys)

    def __getitem__(self, key: str):
        return self.model_extra[key] if self.model_extra and key in self.model_extra else None

    def children_of(self, key: str) -> list['Element']:
        if key not in self.children_keys:
            raise ChildrenKeyNotFoundError(key, self.aspect)
        return self.model_extra.get(key, []) if self.model_extra else []

    def element_dict(self):
        """Returns a dictionary composed of id and props"""
        return {"id": self.id, "uri": self.uri, **self.props()}
    
    def context_dict(self):
        """Returns a dictionary composed of id, uri, relations, props and children_ids"""
        return {
            **self.element_dict(),
            "relations": self.relations,
            "aspect": self.aspect,
            **self.children_ids(),
        }

    def element_str(self):
        return json.dumps(self.element_dict(), ensure_ascii=False, sort_keys=True)

    def children_dict(self):
        """Returns id + props + children (children elements are child-level element_dict results)"""
        data = self.element_dict()
        for key in self.children_keys:
            children = self.children_of(key)
            data[key] = [child.context_dict() for child in children]
        return data

    def nested_dict(self):
        """Returns id + props + children (children elements recursively call nested_dict)"""
        data = self.element_dict()
        for key in self.children_keys:
            children = self.children_of(key)
            data[key] = [child.nested_dict() for child in children]
        return data

    def dumped_dict(self):
        """Returns id + relations + props + children (children elements recursively call dumped_dict)"""
        data = {"id": self.id, "relations": self.relations, **self.props()}
        for key in self.children_keys:
            children = self.children_of(key)
            data[key] = [child.dumped_dict() for child in children]
        return data

    def update(self, props: dict[str, Any]):
        undo = {}
        for k, v in props.items():
            if k in ['id', 'uri', 'relations', 'aspect', 'children_keys', 'embedding', 'hash']:
                logger.warning(f'Ignore Private Property "{k}" Update.')
            elif k in self.children_keys:
                logger.warning(f'Ignore Children Key "{k}" Update.')
            elif self.model_extra is None:
                logger.warning(f'Ignore Update for Element with no model_extra: {self.uri}')
            elif k in self.model_extra and v is None:
                undo[k] = self.model_extra[k]
                del self.model_extra[k]
            elif v is not None:
                self.model_extra[k] = v
                undo[k] = None
        return undo

    def update_children(self, key: str, children: list['Element']):
        if self.model_extra is not None:
            self.model_extra[key] = children
        else:
            logger.warning(f'Ignore Update for Element with no model_extra: {self.uri}')

    def update_relations(self, target_uri: str, relations: list[str]):
        old = self.relations.get(target_uri, [])
        self.relations[target_uri] = relations
        return old


@dataclass
class DirectiveElement:
    inner: Element
    parent: Optional['DirectiveElement']
    prev: Optional['DirectiveElement']
    next: Optional['DirectiveElement']
    children: dict[str, 'DirectiveElementList']

    @classmethod
    def wrap(cls, ele: Element, children_keys: list[str], *, parent: Optional['DirectiveElement'] = None):
        wrapped = cls(inner=ele, parent=parent, prev=None, next=None, children={})
        for key in children_keys:
            if ele.model_extra and key in ele.model_extra and isinstance(ele.model_extra[key], list):
                wrapped.children[key] = DirectiveElementList.wrap(ele.model_extra[key], children_keys, parent=wrapped)
        return wrapped

    @staticmethod
    def wrap_list(elements: list[Element], children_keys: list[str], *, parent: Optional['DirectiveElement'] = None):
        wrapped = [DirectiveElement.wrap(ele, children_keys, parent=parent) for ele in elements]
        for idx, ele in enumerate(wrapped):
            if idx > 0:
                ele.prev = wrapped[idx - 1]
            if idx < len(wrapped) - 1:
                ele.next = wrapped[idx + 1]
        return wrapped

    @property
    def props(self):
        return self.inner.props()

    @property
    def id(self):
        return str(self.inner.id)
    
    @property
    def uri(self):
        return self.inner.uri

    @property
    def aspect(self):
        return self.inner.aspect

    @property
    def flattened_child_ids(self):
        return self.inner.flattened_child_ids()

    @property
    def element_dict(self):
        """Returns a dictionary composed of id and props"""
        return self.inner.element_dict()
    
    @property
    def context_dict(self):
        """Returns a dictionary composed of id, uri, relations, props and children_ids"""
        return self.inner.context_dict()

    @property
    def children_dict(self):
        """Returns id + props + children (children elements are child-level element_dict results)"""
        return self.inner.children_dict()

    @property
    def nested_dict(self):
        """Returns id + props + children (children elements recursively call nested_dict)"""
        return self.inner.nested_dict()

    @property
    def relations(self):
        return self.inner.relations

    def __getitem__(self, key: str):
        return self.inner[key]

    def children_of(self, key: str):
        if key not in self.inner.children_keys:
            raise ChildrenKeyNotFoundError(key, self.inner.aspect)
        return self.children.get(key, DirectiveElementList())

    def update(self, props: dict[str, Any]):
        return self.inner.update(props)

    def update_relations(self, target_uri: str, relations: list[str]):
        return self.inner.update_relations(target_uri, relations)

    def splice_at(self, children_key: str, start: int, end: int, *items: 'Element') -> tuple[list['DirectiveElement'], list['DirectiveElement']]:
        old = self.children_of(children_key)[start: end]
        new = DirectiveElementList.wrap(list(items), self.inner.children_keys, parent=self)
        new_list = self.children_of(children_key).splice(start, end, *new)
        self.children[children_key] = new_list
        self.inner.update_children(children_key, [ele.inner for ele in new_list])
        return new, old


class DirectiveElementList(list[DirectiveElement]):
    def __init__(self, children: list[DirectiveElement] | None = None):
        super().__init__(children or [])
        self._ensure_link()

    def _ensure_link(self):
        for idx, ele in enumerate(self):
            if idx > 0:
                ele.prev = self[idx - 1]
            if idx < len(self) - 1:
                ele.next = self[idx + 1]

    @classmethod
    def wrap(cls, elements: list[Element], children_keys: list[str], *, parent: Optional['DirectiveElement'] = None):
        wrapped = [DirectiveElement.wrap(ele, children_keys, parent=parent) for ele in elements]
        wrapped = cls(wrapped)
        return wrapped

    def splice(self, start: int, end: int, *items: DirectiveElement):
        result = self[:start] + list(items) + self[end:]
        return DirectiveElementList(result)
