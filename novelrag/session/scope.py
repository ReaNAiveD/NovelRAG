from abc import ABC, abstractmethod

from novelrag.config.intent import IntentConfig
from novelrag.exceptions import AspectNotFoundError
from novelrag.intent import IntentFactory, DictionaryIntentFactory
from novelrag.resource import ResourceAspect, ResourceRepository


class IntentScope:
    def __init__(self, name: str, data: ResourceAspect | None, intents: IntentFactory):
        self.name = name
        self.data = data
        self.intents = intents


class IntentScopeFactory(ABC):
    @abstractmethod
    async def get(self, name: str) -> IntentScope:
        raise NotImplementedError()


class ConfigBasedIntentScopeFactory(IntentScopeFactory):
    def __init__(
            self,
            *,
            resource_repository: ResourceRepository,
            scope_configurations: dict[str, dict[str, IntentConfig]],
    ):
        self.resource_repository = resource_repository
        self.intent_configurations = scope_configurations

    async def get(self, name: str) -> IntentScope:
        data = await self.resource_repository.get_aspect(name)
        registry = DictionaryIntentFactory.from_config(self.intent_configurations.get(name, {}))
        return IntentScope(name, data, registry)
