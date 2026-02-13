import os
from enum import Enum

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated, Literal


class ChatLLMType(str, Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"
    DeepSeek = "deepseek"


class OpenAIChatConfig(BaseModel):
    """Configuration for OpenAI-compatible chat endpoints (maps to ``ChatOpenAI``)."""

    type: Literal[ChatLLMType.OpenAI]
    endpoint: Annotated[str, Field(
        description="The OpenAI-compatible base URL",
    )]
    api_key: Annotated[str, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ["OPENAI_API_KEY"],
    )]
    timeout: Annotated[float, Field(
        description="Request timeout in seconds",
        default=180.0,
    )]
    model: Annotated[str, Field(
        description="The model identifier to use for chat completions",
    )]
    json_supports: Annotated[bool, Field(
        description="Whether the model supports JSON output mode.",
        default=False,
    )]

    max_tokens: Annotated[int | None, Field(
        description="The maximum number of tokens to generate in the response",
        default=4000,
    )]
    temperature: Annotated[float | None, Field(
        description="Controls randomness in the model's output (0.0 to 2.0)",
        default=0.0,
    )]
    top_p: Annotated[float | None, Field(
        description="Controls diversity via nucleus sampling (0.0 to 1.0)",
        default=1.0,
    )]
    n: Annotated[int | None, Field(
        description="Number of chat completion choices to generate",
        default=1,
    )]
    frequency_penalty: Annotated[float | None, Field(
        description="Penalizes repeated tokens (-2.0 to 2.0)",
        default=0.0,
    )]
    presence_penalty: Annotated[float | None, Field(
        description="Penalizes repeated topics (-2.0 to 2.0)",
        default=0.0,
    )]

    def langchain_kwargs(self) -> dict:
        """Return kwargs suitable for ``ChatOpenAI(...)``."""
        return {
            "model": self.model,
            "base_url": self.endpoint,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class AzureOpenAIChatConfig(OpenAIChatConfig):
    """Configuration for Azure OpenAI chat (maps to ``AzureChatOpenAI``)."""

    type: Literal[ChatLLMType.AzureOpenAI]
    deployment: Annotated[str, Field(
        description="The deployment name for the chat model",
    )]
    api_key: Annotated[str | None, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
    )]
    api_version: Annotated[str, Field(
        description="The Azure OpenAI API version to use",
    )]

    def langchain_kwargs(self) -> dict:
        """Return kwargs suitable for ``AzureChatOpenAI(...)``."""
        kw: dict = {
            "model": self.model,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.api_key:
            kw["api_key"] = self.api_key
        else:
            # Use Azure AD token-based auth
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            kw["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
        return kw


class DeepSeekChatConfig(OpenAIChatConfig):
    """Configuration for DeepSeek chat (uses ``ChatOpenAI`` with DeepSeek base URL)."""

    type: Literal[ChatLLMType.DeepSeek]
    api_key: Annotated[str, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ["DEEPSEEK_API_KEY"],
    )]
    endpoint: Annotated[str, Field(
        description="The DeepSeek endpoint URL",
        default="https://api.deepseek.com",
    )]


ChatConfig = Annotated[
    AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig,
    Field(
        description="Configuration for the chat completion model",
        discriminator="type",
    ),
]


def validate_chat_config(data: dict) -> ChatConfig:
    """Validate and return a ChatConfig instance from raw data."""
    return TypeAdapter(ChatConfig).validate_python(data)


class EmbeddingLLMType(str, Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"


class OpenAIEmbeddingConfig(BaseModel):
    """Configuration for OpenAI-compatible embeddings (maps to ``OpenAIEmbeddings``)."""

    type: Literal[EmbeddingLLMType.OpenAI]
    endpoint: Annotated[str, Field(
        description="The OpenAI-compatible base URL",
    )]
    api_key: Annotated[str | None, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
    )]
    timeout: Annotated[float, Field(
        description="Request timeout in seconds",
        default=180.0,
    )]
    model: Annotated[str, Field(
        description="The model identifier to use for embeddings",
    )]

    def langchain_kwargs(self) -> dict:
        """Return kwargs suitable for ``OpenAIEmbeddings(...)``."""
        return {
            "model": self.model,
            "base_url": self.endpoint,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }


class AzureOpenAIEmbeddingConfig(BaseModel):
    """Configuration for Azure OpenAI embeddings (maps to ``AzureOpenAIEmbeddings``)."""

    type: Literal[EmbeddingLLMType.AzureOpenAI]
    endpoint: Annotated[str, Field(
        description="The Azure OpenAI endpoint URL",
    )]
    deployment: Annotated[str, Field(
        description="The deployment name for the embedding model",
    )]
    api_version: Annotated[str, Field(
        description="The Azure OpenAI API version to use",
    )]
    api_key: Annotated[str | None, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
    )]
    timeout: Annotated[float, Field(
        description="Request timeout in seconds",
        default=180.0,
    )]
    model: Annotated[str, Field(
        description="The model identifier to use for embeddings",
    )]

    def langchain_kwargs(self) -> dict:
        """Return kwargs suitable for ``AzureOpenAIEmbeddings(...)``."""
        kw: dict = {
            "model": self.model,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment,
            "api_version": self.api_version,
            "timeout": self.timeout,
        }
        if self.api_key:
            kw["api_key"] = self.api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            kw["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
        return kw


EmbeddingConfig = Annotated[
    OpenAIEmbeddingConfig | AzureOpenAIEmbeddingConfig,
    Field(
        description="Configuration for the embedding model",
        discriminator="type",
    ),
]
