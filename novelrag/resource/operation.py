import re
from enum import Enum
from typing_extensions import Annotated, Any, Literal

from pydantic import BaseModel, Field

from novelrag.resource import Element

# Regular expression to find parts of the path
PATH_REGEX = re.compile(r'\w+|\[\d+]|\["[^"]+"]|\[\w+]')


def split_path(path: str) -> list[str]:
    substitutions = [
        (r'\\', r'\0', '\\'), # Escaped backslash
        (r'\"', r'\1', '\"'), # Escaped double quote
        (r'\'', r'\2', "\'"), # Escaped single quote
        (r'\n', r'\3', '\n'), # New line
        (r'\t', r'\4', '\t'), # Tab
        (r'\r', r'\5', '\r'), # Carriage return
        (r'\b', r'\6', '\b'), # Backspace
        (r'\f', r'\7', '\f'), # Formfeed
        (r'\v', r'\8', '\v'), # Vertical tab
    ]
    for escape_seq, placeholder, _ in substitutions:
        path = path.replace(escape_seq, placeholder)
    # Find all matches
    matches = PATH_REGEX.findall(path)
    # Process matches to remove brackets and quotes
    result = []
    for match in matches:
        if match.startswith('[') and match.endswith(']'):
            # Remove brackets and quotes
            match = match.strip('[]').strip('"').strip("'")
        result.append(match)

    # Restore escaped sequences
    for i, part in enumerate(result):
        for _, escape_seq, actual in substitutions:
            result[i] = result[i].replace(escape_seq, actual)

    return result


class OperationTarget(str, Enum):
    PROPERTY = 'property'
    ELEMENT = 'element'


class PropertyOperation(BaseModel):
    target: Literal[OperationTarget.PROPERTY]
    element_id: Annotated[str, Field(description='Id of the element the operation happens on')]
    data: Annotated[dict[str, Any], Field(description='The data that updates on the element')]

    def create_undo(self, previous: dict[str, Any]):
        return self.model_copy(update={'data': previous})


class OperationLocationType(str, Enum):
    ELEMENT = 'element'
    ASPECT = 'aspect'


class ElementLocation(BaseModel):
    type: Literal[OperationLocationType.ELEMENT]
    element_id: Annotated[str, Field()]
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


Operation = Annotated[PropertyOperation | ElementOperation, Field(discriminator='target')]


class ObjectLocation:
    def __init__(self, parent: dict | list | None, idx: str | None, loc: list[str]):
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


def extract_location(obj: dict | list | str | int | float | bool | None, path: str):
    op_path = split_path(path)
    if not op_path:
        return ObjectLocation(None, None, [])
    parent_obj = obj
    handled_path = []
    for idx, key in enumerate(op_path[:-1]):
        handled_path.append(key)
        if isinstance(parent_obj, dict):
            try:
                parent_obj = parent_obj[key]
            except KeyError as e:
                raise KeyError(f"Key '{key}' of Path {_format_path(handled_path)} not Exists.") from e
        elif isinstance(parent_obj, list):
            try:
                parent_obj = parent_obj[int(key)]
            except IndexError as e:
                raise KeyError(f"Index '{key}' of Path {_format_path(handled_path)} not Exists.") from e
            except ValueError as e:
                raise KeyError(f"Key '{key}' of Path {_format_path(handled_path)} is not Allowed for a list.") from e
        else:
            raise KeyError(f"Path {_format_path(handled_path)} is not a dict or list.")
    return ObjectLocation(parent_obj, op_path[-1] if op_path else None, handled_path)
