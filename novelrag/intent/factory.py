from abc import ABC, abstractmethod
import importlib
from typing import Sequence

from novelrag.config.intent import IntentConfig
from .intent import Intent


def build_intent(name: str, intent_config: IntentConfig) -> Intent:
    name = intent_config.name or name
    pkg, cls_name = intent_config.cls.rsplit('.', maxsplit=1)
    module = importlib.import_module(pkg)
    intent_cls = getattr(module, cls_name)
    intent = intent_cls(name=name, **intent_config.kwargs)
    assert isinstance(intent, Intent)
    return intent


class IntentFactory(ABC):
    @abstractmethod
    async def get_intent(self, name: str | None) -> Intent | None:
        raise NotImplementedError()


class DictionaryIntentFactory(IntentFactory):
    def __init__(self, intents: Sequence[Intent]):
        self.intents = dict((intent.name, intent) for intent in intents)

    @classmethod
    def from_config(cls, config: dict[str, IntentConfig]):
        intents = [build_intent(name, conf) for name, conf in config.items()]
        return cls(intents)

    async def get_intent(self, name: str | None) -> Intent | None:
        if name is None:
            return self.intents.get('_default')
        return self.intents.get(name)
