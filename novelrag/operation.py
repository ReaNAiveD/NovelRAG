import re
from enum import Enum

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Any, Optional


class OperationType(str, Enum):
    NEW = 'new'
    UPDATE = 'update'
    PUT = 'put'
    DELETE = 'delete'


class Operation(BaseModel):
    type: OperationType
    path: Annotated[str, Field()]
    data: Optional[Any]


def apply_operation(obj: dict | list | str | int | float | bool | None, op: Operation) -> tuple[dict | list | str | int | float | bool | None, Operation]:
    op_path = split_path(op.path)
    parent_obj = None
    target_obj = obj
    for idx, key in enumerate(op_path):
        if isinstance(target_obj, dict):
            parent_obj = target_obj
            if idx == len(op_path) - 1 and key not in target_obj:
                target_obj = None
                break
            target_obj = target_obj[key]
        elif isinstance(target_obj, list):
            parent_obj = target_obj
            key_int = int(key)
            if idx == len(op_path) - 1 and key_int >= len(target_obj):
                target_obj = None
                break
            target_obj = target_obj[key_int]
        elif idx == len(op_path) - 1:
            parent_obj = target_obj
            target_obj = None
        else:
            current_path = ''.join([f'[{key}]' if key.isnumeric() else f'["{key}"]' for key in op_path])
            raise Exception(f"Path {current_path} of {obj} is not a dict or list.")
    match op.type:
        case OperationType.NEW:
            if isinstance(parent_obj, dict):
                if op_path[-1] in parent_obj:
                    raise Exception(f"Path {op.path} of {obj} already exists.")
                parent_obj[op_path[-1]] = op.data
                undo = Operation(type=OperationType.DELETE, path=op.path, data=None)
                return obj, undo
            elif isinstance(parent_obj, list):
                parent_obj.insert(int(op_path[-1]), op.data)
                undo = Operation(type=OperationType.DELETE, path=op.path, data=None)
                return obj, undo
            elif not obj:
                undo = Operation(type=OperationType.DELETE, path='', data=None)
                return op.data, undo
            else:
                raise Exception(f"Root of {obj} has been occupied.")
        case OperationType.UPDATE:
            if isinstance(target_obj, dict):
                undo = Operation(type=OperationType.PUT, path=op.path, data=dict(target_obj))
                target_obj.update(op.data)
                return obj, undo
            else:
                raise Exception(f"Update is not a valid operation for {str(type(target_obj))}.")
        case OperationType.PUT:
            if isinstance(parent_obj, dict):
                if op_path[-1] in parent_obj:
                    undo = Operation(type=OperationType.PUT, path=op.path, data=parent_obj[op_path[-1]])
                    parent_obj[op_path[-1]] = op.data
                    return obj, undo
                else:
                    parent_obj[op_path[-1]] = op.data
                    undo = Operation(type=OperationType.DELETE, path=op.path, data=None)
                    return obj, undo
            elif isinstance(parent_obj, list):
                idx = int(op_path[-1])
                if len(parent_obj) == idx:
                    parent_obj.append(op.data)
                    undo = Operation(type=OperationType.DELETE, path=op.path, data=None)
                    return obj, undo
                elif len(parent_obj) > idx:
                    undo = Operation(type=OperationType.PUT, path=op.path, data=parent_obj[idx])
                    parent_obj[idx] = op.data
                    return obj, undo
                else:
                    raise Exception(f"Index {idx} of {obj} is out of range.")
            else:
                undo = Operation(type=OperationType.PUT, path='', data=obj)
                return op.data, undo
        case OperationType.DELETE:
            if isinstance(parent_obj, dict):
                if op_path[-1] in parent_obj:
                    undo = Operation(type=OperationType.NEW, path=op.path, data=parent_obj[op_path[-1]])
                    del parent_obj[op_path[-1]]
                    return obj, undo
                else:
                    raise Exception(f"Key {op_path[-1]} of {parent_obj} is not Existed.")
            elif isinstance(parent_obj, list):
                idx = int(op_path[-1])
                if len(parent_obj) > idx:
                    undo = Operation(type=OperationType.NEW, path=op.path, data=parent_obj[idx])
                    del parent_obj[idx]
                    return obj, undo
                else:
                    raise Exception(f"Index {idx} of {obj} is out of range.")
            else:
                undo = Operation(type=OperationType.NEW, path='', data=obj)
                return None, undo
        case _:
            raise Exception(f"Unrecognized Operation: {str(op.type)}")


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
