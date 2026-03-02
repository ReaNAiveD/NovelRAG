import json
import logging

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from novelrag.exceptions import ChildrenKeyNotFoundError

logger = logging.getLogger(__name__)


class Element(BaseModel):
    """A flat data record belonging to an aspect.

    Children relationships are represented as **ordered lists of names**
    stored in ``model_extra`` under the appropriate ``children_keys``
    entry.  The application layer can reconstruct URIs from these names
    (``{parent_uri}/{child_name}``).
    """

    id: Annotated[str, Field(description='Id of the element')]
    uri: Annotated[str, Field(description='URI of the element')]
    relationships: Annotated[dict[str, list[str]], Field(description='Related Elements. <Id>: <Description>', default_factory=lambda: {})]
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
        """Build a *flat* Element instance from a dictionary value.

        Children entries (keys listed in *children_keys*) are normalised to
        ``list[str]``.
        """
        if 'id' not in value:
            raise ValueError(
                f"Element data under '{parent_uri}' is missing required 'id' field. "
                f"Got keys: {list(value.keys())}. "
                f"Every resource (including items in children_keys lists) must have an 'id' field."
            )
        uri = f'{parent_uri}/{value["id"]}'
        value = dict(value)
        value['uri'] = uri
        value['aspect'] = aspect
        value['children_keys'] = children_keys

        for key in children_keys:
            if key in value and isinstance(value[key], list):
                normalised: list[str] = []
                for item in value[key]:
                    if isinstance(item, dict) and 'id' in item:
                        normalised.append(item['id'])
                    elif isinstance(item, str):
                        normalised.append(item)
                value[key] = normalised

        return cls.model_validate(value, strict=strict, from_attributes=from_attributes, context=context)

    # -- Property access -----------------------------------------------------

    @property
    def props(self):
        """
        Returns a dictionary of properties excluding children keys.
        Usually includes all properties in model_extra except those defined in children_keys.
        Excludes properties like 'id', 'relationships', 'aspect', and 'children_keys'.
        """
        return dict((k, v) for k, v in self.model_extra.items() if k not in self.children_keys) if self.model_extra else {}

    @property
    def children_names(self) -> dict[str, list[str]]:
        """Return ``{children_key: [child_name, …]}`` for every children key."""
        if not self.model_extra:
            return {key: [] for key in self.children_keys}
        return {
            key: list(self.model_extra.get(key, []))
            for key in self.children_keys
        }

    def children_names_of(self, key: str) -> list[str]:
        """Return the ordered list of child names for *key*."""
        if key not in self.children_keys:
            raise ChildrenKeyNotFoundError(key, self.aspect)
        if not self.model_extra:
            return []
        return list(self.model_extra.get(key, []))

    def __getitem__(self, key: str):
        return self.model_extra[key] if self.model_extra and key in self.model_extra else None

    # -- Dict representations -----------------------------------------------

    @property
    def element_dict(self):
        """Returns a dictionary composed of id, uri and props (no children)."""
        return {"id": self.id, "uri": self.uri, **self.props}

    @property
    def context_dict(self):
        """Returns a dictionary composed of id, uri, relationships, props and children names."""
        return {
            **self.element_dict,
            "relationships": self.relationships,
            "aspect": self.aspect,
            **self.children_names,
        }

    def element_str(self):
        return json.dumps(self.element_dict, ensure_ascii=False, sort_keys=True)

    def dumped_dict(self):
        """Serialisation-ready dict: id + relationships + props + children names."""
        return {"id": self.id, "relationships": self.relationships, **self.props, **self.children_names}

    # -- Mutations -----------------------------------------------------------

    def update(self, props: dict[str, Any]):
        """Update properties on this element.

        Returns a dict of previous values suitable for undo.  Children-key
        fields, core fields (``id``, ``uri``, …) and embedding metadata are
        silently skipped.
        """
        undo: dict[str, Any] = {}
        for k, v in props.items():
            if k in ['id', 'uri', 'relationships', 'aspect', 'children_keys', 'embedding', 'hash']:
                logger.warning(f'Ignore Private Property "{k}" Update.')
            elif k in self.children_keys:
                logger.warning(f'Ignore Children Key "{k}" Update.')
            elif self.model_extra is None:
                logger.warning(f'Ignore Update for Element with no model_extra: {self.uri}')
            elif k in self.model_extra and v is None:
                undo[k] = self.model_extra[k]
                del self.model_extra[k]
            elif v is not None:
                undo[k] = self.model_extra.get(k)
                self.model_extra[k] = v
        return undo

    def update_relationships(self, target_uri: str, relationships: list[str]):
        old = self.relationships.get(target_uri, [])
        self.relationships[target_uri] = relationships
        return old

    def set_children_names(self, key: str, names: list[str]):
        """Replace the ordered children-name list for *key*."""
        if key not in self.children_keys:
            raise ChildrenKeyNotFoundError(key, self.aspect)
        if self.model_extra is not None:
            self.model_extra[key] = names
        else:
            logger.warning(f'Ignore set_children_names for Element with no model_extra: {self.uri}')

    def add_child_names(self, key: str, names: list[str]):
        """Add to the ordered children-name list for *key*."""
        if key not in self.children_keys:
            raise ChildrenKeyNotFoundError(key, self.aspect)
        if self.model_extra is not None:
            existing = self.model_extra.get(key, [])
            self.model_extra[key] = existing + names
        else:
            logger.warning(f'Ignore add_child_names for Element with no model_extra: {self.uri}')
