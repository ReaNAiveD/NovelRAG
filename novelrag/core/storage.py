import os.path

import yaml
from typing import TypeVar, Generic
from pydantic import BaseModel

from novelrag.core.registry import model_registry
from novelrag.core.config import NovelStorageConfig
from novelrag.core.operation import Operation, apply_operation
from novelrag.model import Premise

T = TypeVar('T', bound=BaseModel)

class AspectStorage(Generic[T]):
    """Manages loading and saving of aspect data that conforms to Pydantic models"""
    def __init__(self, file_path: str, model_class: type[T]):
        self.file_path = file_path
        self.model_class = model_class
        self._cached_data = None

    @property
    def data(self) -> T:
        if self._cached_data is None:
            self._cached_data = self.load()
        return self._cached_data

    @data.setter
    def data(self, value: T):
        if not isinstance(value, self.model_class):
            raise TypeError(f"Expected {self.model_class.__name__}, got {type(value).__name__}")
        self.save(value)
        self._cached_data = value

    def load(self) -> T | None:
        if not os.path.exists(self.file_path):
            return self.model_class()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
        return self.model_class.model_validate(raw_data)

    def save(self, data: T):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data.model_dump(), f, allow_unicode=True)

    def apply(self, op: Operation) -> Operation:
        new_data, undo = apply_operation(self.data.model_dump(), op)
        self.data = self.model_class.model_validate(new_data)
        return undo


class NovelStorage:
    def __init__(self, config: NovelStorageConfig):
        self.config = config
        self.storages = {}

    def __getitem__(self, item: str):
        if item in self.storages:
            return self.storages[item]
        storage_config = self.config[item]
        model_class = model_registry[storage_config.model]
        storage = AspectStorage(storage_config.file_path, model_class)
        self.storages[item] = storage
        return storage

    def premise(self) -> AspectStorage[Premise]:
        return self['premise']
