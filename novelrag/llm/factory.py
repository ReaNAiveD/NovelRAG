import importlib.util

from novelrag.config.llm import ChatConfig, AzureOpenAIChatConfig, EmbeddingConfig, AzureOpenAIEmbeddingConfig, \
    OpenAIChatConfig, DeepSeekChatConfig
from novelrag.exceptions import NoChatLLMConfigError, NoEmbeddingConfigError
from .types import ChatLLM, EmbeddingLLM


class ChatLLMFactory:
    def __init__(self, default: ChatLLM | None = None):
        self.default = default

    @classmethod
    def build(cls, config: ChatConfig) -> ChatLLM:
        if importlib.util.find_spec('openai'):
            if isinstance(config, AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig):
                from .oai import OpenAIChatLLM
                return OpenAIChatLLM.from_config(config)
            else:
                raise Exception(f'Unexpected Config: {config}')
        elif importlib.util.find_spec('azure.ai.inference'):
            if isinstance(config, AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig):
                from .azure_ai import AzureAIChatLLM
                return AzureAIChatLLM.from_config(config)
            else:
                raise Exception(f'Unexpected Config: {config}')
        else:
            raise Exception(f'No AI SDK found: please install openai or azure-ai-inference')

    def get(self, config: ChatConfig | None = None) -> ChatLLM:
        if config:
            return self.build(config)
        if self.default:
            return self.default
        raise NoChatLLMConfigError()


class EmbeddingLLMFactory:
    def __init__(self, default: EmbeddingLLM | None = None):
        self.default = default

    @classmethod
    def build(cls, config: EmbeddingConfig) -> EmbeddingLLM:
        if importlib.util.find_spec('openai'):
            if isinstance(config, AzureOpenAIEmbeddingConfig):
                from .oai import OpenAIEmbeddingLLM
                return OpenAIEmbeddingLLM.from_config(config)
            else:
                raise Exception(f'Unexpected Config: {config}')
        elif importlib.util.find_spec('azure.ai.inference'):
            if isinstance(config, AzureOpenAIEmbeddingConfig):
                from .azure_ai import AzureAIEmbeddingLLM
                return AzureAIEmbeddingLLM.from_config(config)
            else:
                raise Exception(f'Unexpected Config: {config}')
        else:
            raise Exception(f'No AI SDK found: please install openai or azure-ai-inference')

    def get(self, config: EmbeddingConfig | None = None) -> EmbeddingLLM:
        if config:
            return self.build(config)
        elif self.default:
            return self.default
        raise NoEmbeddingConfigError()
