from enum import Enum
from typing_extensions import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class OperationTarget(str, Enum):
    PROPERTY = 'property'
    ELEMENT = 'element'


class PropertyOperation(BaseModel):
    target: Literal[OperationTarget.PROPERTY]
    element_uri: Annotated[str, Field(description='Uri of the element the operation happens on')]
    data: Annotated[dict[str, Any], Field(description='The data that updates on the element')]

    @classmethod
    def new(cls, element_uri: str, data: dict[str, Any]):
        return cls(target=OperationTarget.PROPERTY, element_uri=element_uri, data=data)

    def create_undo(self, undo_update: dict[str, Any]):
        return self.model_copy(update={'data': undo_update})


class OperationLocationType(str, Enum):
    ELEMENT = 'element'
    ASPECT = 'aspect'


class ElementLocation(BaseModel):
    type: Literal[OperationLocationType.ELEMENT]
    element_uri: Annotated[str, Field()]
    children_key: Annotated[str, Field()]


class AspectLocation(BaseModel):
    type: Literal[OperationLocationType.ASPECT]
    aspect: Annotated[str, Field()]

    @classmethod
    def new(cls, aspect: str):
        return cls(type=OperationLocationType.ASPECT, aspect=aspect)


OperationLocation = ElementLocation | AspectLocation


class ElementOperation(BaseModel):
    target: Literal[OperationTarget.ELEMENT]
    location: Annotated[OperationLocation, Field(discriminator='type')]
    start: Annotated[int, Field(description='Start Index of Splice')]
    end: Annotated[int, Field(description='End Index of Splice')]
    data: Annotated[list[dict[str, Any]] | None, Field(description='The data that splice on the list.')]

    @classmethod
    def new(cls, location: OperationLocation, *, start: int=0, end: int | None = None, data: list[dict[str, Any]] | None = None):
        return cls(
            target=OperationTarget.ELEMENT,
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


Operation = Annotated[PropertyOperation | ElementOperation, Field(discriminator='target')]


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
