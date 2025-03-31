from abc import ABC, abstractmethod
from dataclasses import dataclass

from .action import Action
from novelrag.pending_queue import PendingUpdateQueue
from novelrag.resource import ResourceRepository, DirectiveElement, ResourceAspect
from novelrag.llm import EmbeddingLLMFactory, ChatLLMFactory, ChatLLM
from novelrag.exceptions import IntentMissingNameError
from novelrag.config.llm import ChatConfig
from novelrag.conversation import ConversationHistory


@dataclass
class IntentContext:
    chat_llm_factory: ChatLLMFactory
    embedding_factory: EmbeddingLLMFactory
    current_element: DirectiveElement | None = None
    current_aspect: ResourceAspect | None = None
    raw_command: str | None = None
    pending_update: PendingUpdateQueue | None = None
    resource_repository: ResourceRepository | None = None
    conversation_history: ConversationHistory | None = None

    @property
    def aspect_name(self) -> str | None:
        return self.current_aspect.name if self.current_aspect else None


class Intent(ABC):
    def __init__(self, *, name: str | None, **kwargs):
        self._name = name

    @property
    @abstractmethod
    def default_name(self) -> str | None:
        return None

    @property
    def name(self):
        name = self._name or self.default_name
        if not name:
            raise IntentMissingNameError()
        return name

    @abstractmethod
    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        pass


class LLMIntent(Intent, ABC):
    def __init__(self, *, name: str | None, chat_llm: dict | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.chat_llm_config = chat_llm
        self._chat_llm: ChatLLM | None = None

    def chat_llm(self, factory: ChatLLMFactory):
        if not self._chat_llm:
            chat_config = ChatConfig.model_validate(self.chat_llm_config) if self.chat_llm_config else None
            self._chat_llm = factory.get(chat_config)
        return self._chat_llm
