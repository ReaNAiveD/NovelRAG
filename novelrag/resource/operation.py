from enum import Enum
from typing_extensions import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class OperationTarget(str, Enum):
    PROPERTY = 'property'
    RESOURCE = 'resource'


class PropertyOperation(BaseModel):
    target: Literal[OperationTarget.PROPERTY]
    resource_uri: Annotated[str, Field(description='Uri of the resource the operation happens on')]
    data: Annotated[dict[str, Any], Field(description='The data that updates on the resource')]

    @classmethod
    def new(cls, resource_uri: str, data: dict[str, Any]):
        return cls(target=OperationTarget.PROPERTY, resource_uri=resource_uri, data=data)

    def create_undo(self, undo_update: dict[str, Any]):
        return self.model_copy(update={'data': undo_update})


class ResourceLocation(BaseModel):
    """Location for resource operations using URI-based addressing.
    
    Can point to either:
    - An aspect's root list: resource_uri="/character", children_key=None
    - A resource's children list: resource_uri="/outline/chapter_1", children_key="scenes"
    """
    resource_uri: Annotated[str, Field(description='URI of the resource or aspect')]
    children_key: Annotated[str | None, Field(description='Key for nested children list, None for aspect root')]

    @classmethod
    def aspect(cls, aspect: str):
        """Create a location pointing to an aspect's root list."""
        return cls(resource_uri=f"/{aspect}", children_key=None)
    
    @classmethod
    def resource(cls, resource_uri: str, children_key: str):
        """Create a location pointing to a resource's children list."""
        return cls(resource_uri=resource_uri, children_key=children_key)


class ResourceOperation(BaseModel):
    target: Literal[OperationTarget.RESOURCE]
    location: Annotated[ResourceLocation, Field(description='Location where the operation occurs')]
    start: Annotated[int, Field(description='Start Index of Splice')]
    end: Annotated[int, Field(description='End Index of Splice')]
    data: Annotated[list[dict[str, Any]] | None, Field(description='The data that splice on the list.')]

    @classmethod
    def new(cls, location: ResourceLocation, *, start: int=0, end: int | None = None, data: list[dict[str, Any]] | None = None):
        return cls(
            target=OperationTarget.RESOURCE,
            location=location,
            start=start,
            end=end if end is not None else start,
            data=data,
        )

    def create_undo(self, previous: list[dict[str, Any]]):
        assert len(previous) == self.end - self.start
        return self.new(
            self.location.model_copy(),
            start=self.start,
            end=self.start + len(self.data) if self.data else self.start,
            data=previous,
        )


Operation = Annotated[PropertyOperation | ResourceOperation, Field(discriminator='target')]


def validate_op(op: dict) -> Operation: # type: ignore
    return TypeAdapter(Operation).validate_python(op) # type: ignore


def validate_op_json(op: str) -> Operation: # type: ignore
    return TypeAdapter(Operation).validate_json(op) # type: ignore


class ObjectLocation:
    def __init__(self, parent: dict[str, Any] | list | None, idx: str | None, loc: list[str]): # type: ignore
        self.parent = parent
        self.idx = idx
        self.loc = loc

    @property
    def target(self):
        if not self.parent or not self.idx:
            return None
        if isinstance(self.parent, dict):
            try:
                return self.parent[self.idx]
            except KeyError as e:
                raise KeyError(f"Key '{self.idx}' of Path {_format_path(self.loc + [self.idx])} not Exists.") from e
        elif isinstance(self.parent, list):
            try:
                return self.parent[int(self.idx)]
            except IndexError as e:
                raise KeyError(f"Index '{self.idx}' of Path {_format_path(self.loc + [self.idx])} not Exists.") from e
            except ValueError as e:
                raise KeyError(f"Key '{self.idx}' of Path {_format_path(self.loc + [self.idx])} is not Allowed for a list.") from e
        else:
            raise KeyError(f"Path {_format_path(self.loc + [self.idx])} is not a dict or list.")


def _format_path(path: list[str]):
    return ''.join([f'[{key}]' if key.isnumeric() else f'["{key}"]' for key in path])
