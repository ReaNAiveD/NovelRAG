import asyncio
import json
import logging
import uuid

from dataclasses import dataclass
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, Field, UUID4
from typing_extensions import Annotated

from novelrag.exceptions import ChildrenKeyNotFoundError
from novelrag.llm.oai.embedding import OpenAIEmbeddingLLM

logger = logging.getLogger(__name__)


class Element(BaseModel):
    id: Annotated[UUID4, Field(description='Id of the element')]
    relations: Annotated[dict[str, str], Field(description='Related Elements. <Id>: <Description>', default_factory=lambda: {})]
    aspect: Annotated[str, Field(description='Aspect of the element')]
    children_keys: Annotated[list[str], Field(default_factory=lambda: [])]
    embedding: Annotated[list[float] | None, Field(default=None)]

    model_config = ConfigDict(extra='allow')

    @classmethod
    def build(
            cls,
            value: dict,
            aspect: str,
            children_keys: list[str],
            *,
            strict: bool | None = None,
            from_attributes: bool | None = None,
            context: Any | None = None
    ):
        if not 'id' in value:
            value['id'] = uuid.uuid4()
        value['aspect'] = aspect
        value['children_keys'] = children_keys
        ele = cls.model_validate(value, strict=strict, from_attributes=from_attributes, context=context)
        for key in children_keys:
            if ele.model_extra and key in ele.model_extra and isinstance(ele.model_extra[key], list):
                ele.model_extra[key] = [cls.build(child, aspect, children_keys, strict=strict, from_attributes=from_attributes, context=context) for child in ele.model_extra[key]]
        return ele

    def props(self):
        return dict((k, v) for k, v in self.model_extra.items() if k not in self.children_keys)

    def children_of(self, key: str):
        if key not in self.children_keys:
            raise ChildrenKeyNotFoundError(key, self.aspect)
        return self.model_extra.get(key, [])

    def update(self, props: dict[str, Any]):
        undo = {}
        for k, v in props.items():
            if k in ['id', 'relations', 'aspect', 'children_keys', 'embedding']:
                logger.warning(f'Ignore Private Property "{k}" Update.')
            elif k in self.children_keys:
                logger.warning(f'Ignore Children Key "{k}" Update.')
            elif k in self.model_extra and v is None:
                undo[k] = self.model_extra[k]
                del self.model_extra[k]
            elif v is not None:
                self.model_extra[k] = v
                undo[k] = None
        return undo

    def update_relations(self, rel: dict[str, str]):
        old = self.relations
        self.relations = rel
        return old

    async def ensure_embedding(self, embedder: OpenAIEmbeddingLLM):
        if self.embedding:
            return
        await self.update_embedding(embedder)

    async def update_embedding(self, embedder: OpenAIEmbeddingLLM):
        old = self.embedding
        data = json.dumps(self.props(), sort_keys=True)
        embeddings = await embedder.embedding(data)
        self.embedding = embeddings[0]
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
                wrapped.children[key] = DirectiveElementList.wrap(ele.model_extra[key], children_keys, parent=parent)
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
    def relations(self):
        return self.inner.relations

    @property
    def embedding(self):
        return self.inner.embedding

    def children_of(self, key: str):
        if key not in self.inner.children_keys:
            raise ChildrenKeyNotFoundError(key, self.inner.aspect)
        return self.children.get(key, [])

    def update(self, props: dict[str, Any]):
        return self.inner.update(props)

    def update_relations(self, rel: dict[str, str]):
        return self.inner.update_relations(rel)

    def splice_at(self, children_key: str, start: int, end: int, *items: 'Element') -> tuple[list['DirectiveElement'], list['DirectiveElement']]:
        old = self.children_of(children_key)[start: end]
        new = DirectiveElementList.wrap(list(items), self.inner.children_keys, parent=self)
        self.children_of(children_key).splice(start, end, *new)
        return new, old

    async def ensure_embedding(self, embedder: OpenAIEmbeddingLLM, *, ensure_children=True):
        await self.inner.ensure_embedding(embedder)
        if ensure_children:
            tasks = [asyncio.create_task(elements.ensure_embeddings(embedder)) for elements in self.children.values()]
            for task in asyncio.as_completed(tasks):
                await task

    async def update_embedding(self, embedder: OpenAIEmbeddingLLM):
        return await self.inner.update_embedding(embedder)


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

    async def ensure_embeddings(self, embedder: OpenAIEmbeddingLLM):
        tasks = [asyncio.create_task(element.ensure_embedding(embedder)) for element in self]
        for task in asyncio.as_completed(tasks):
            await task
