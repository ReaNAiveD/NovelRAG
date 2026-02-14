import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from novelrag.tracer.context import get_active_tracer
from novelrag.config.llm import (
    AzureOpenAIChatConfig,
    AzureOpenAIEmbeddingConfig,
    ChatConfig,
    DeepSeekChatConfig,
    EmbeddingConfig,
    OpenAIChatConfig,
    OpenAIEmbeddingConfig,
)
from novelrag.exceptions import NoChatLLMConfigError, NoEmbeddingConfigError


class ChatLLMFactory:
    def __init__(self, default: BaseChatModel | None = None):
        self.default = default

    @classmethod
    def build(cls, config: ChatConfig) -> BaseChatModel:
        """Build a LangChain ``BaseChatModel`` from a config."""
        if isinstance(config, AzureOpenAIChatConfig):
            from langchain_openai import AzureChatOpenAI
            model = AzureChatOpenAI(**config.langchain_kwargs())
        elif isinstance(config, (OpenAIChatConfig, DeepSeekChatConfig)):
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(**config.langchain_kwargs())
        else:
            raise ValueError(f"Unsupported chat config type: {type(config)}")

        # Attach json_supports flag so downstream code (call_llm_template) can detect it
        object.__setattr__(model, '_json_supports', config.json_supports)

        # Inject tracer callback handler so every ainvoke is auto-traced.
        # Set callbacks directly on the model instance (not via with_config)
        # so they survive .with_structured_output() which wraps the model
        # in a RunnableSequence that would lose RunnableBinding config.
        tracer = get_active_tracer()
        if tracer is not None:
            model.callbacks = [tracer.callback_handler]

        return model

    def get(self, config: ChatConfig | None = None) -> BaseChatModel:
        if config:
            return self.build(config)
        if self.default:
            return self.default
        raise NoChatLLMConfigError()


class EmbeddingLLMFactory:
    def __init__(self, default: Embeddings | None = None):
        self.default = default

    @classmethod
    def build(cls, config: EmbeddingConfig) -> Embeddings:
        """Build a LangChain ``Embeddings`` from a config."""
        if isinstance(config, AzureOpenAIEmbeddingConfig):
            from langchain_openai import AzureOpenAIEmbeddings
            model = AzureOpenAIEmbeddings(**config.langchain_kwargs())
        elif isinstance(config, OpenAIEmbeddingConfig):
            from langchain_openai import OpenAIEmbeddings
            model = OpenAIEmbeddings(**config.langchain_kwargs())
        else:
            raise ValueError(f"Unsupported embedding config type: {type(config)}")

        return model

    def get(self, config: EmbeddingConfig | None = None) -> Embeddings:
        if config:
            return self.build(config)
        elif self.default:
            return self.default
        raise NoEmbeddingConfigError()
