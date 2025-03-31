from abc import ABC, abstractmethod

from novelrag.config.intent import IntentConfig
from novelrag.exceptions import AspectNotFoundError
from novelrag.intent import IntentFactory, DictionaryIntentFactory
from novelrag.resource import ResourceAspect, ResourceRepository


class Aspect:
    def __init__(self, name: str, data: ResourceAspect, intents: IntentFactory):
        self.name = name
        self.data = data
        self.intents = intents


class AspectFactory(ABC):
    @abstractmethod
    async def get(self, name: str) -> Aspect:
        raise NotImplementedError()


class ConfigBasedAspectFactory(AspectFactory):
    def __init__(
            self,
            *,
            resource_repository: ResourceRepository,
            intent_configurations: dict[str, dict[str, IntentConfig]],
    ):
        self.resource_repository = resource_repository
        self.intent_configurations = intent_configurations

    async def get(self, name: str) -> Aspect:
        data = await self.resource_repository.get_aspect(name)
        if not data:
            raise AspectNotFoundError(name, list(self.intent_configurations.keys()))
        registry = DictionaryIntentFactory.from_config(self.intent_configurations.get(name, {}))
        return Aspect(name, data, registry)
