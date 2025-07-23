import os
from enum import Enum

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated, Literal


class ChatLLMType(str, Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"
    DeepSeek = "deepseek"


class OpenAIChatConfig(BaseModel):
    type: Literal[ChatLLMType.OpenAI]
    endpoint: Annotated[str, Field(
        description="The OpenAI endpoint URL",
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

    def chat_params_oai(self) -> dict:
        """Build kwargs for chat completion API calls.

        Returns:
            Dictionary of parameters for chat completion
        """
        return {
            'max_completion_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
        }

    def chat_params(self) -> dict:
        """Build kwargs for chat completion API calls.

        Returns:
            Dictionary of parameters for chat completion
        """
        return {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
        }


class AzureOpenAIChatConfig(OpenAIChatConfig):
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


class DeepSeekChatConfig(OpenAIChatConfig):
    type: Literal[ChatLLMType.DeepSeek]
    api_key: Annotated[str, Field(
        description="The API key for authentication",
        default_factory=lambda: os.environ["DEEPSEEK_API_KEY"],
    )]
    endpoint: Annotated[str, Field(
        description="The DeepSeek endpoint URL",
        default="https://api.deepseek.com",
    )]


ChatConfig = Annotated[AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig, Field(
    description="Configuration for the chat completion model",
    discriminator="type",
)]

def validate_chat_config(data: dict) -> ChatConfig:
    """Validate and return a ChatConfig instance from raw data."""
    return TypeAdapter(ChatConfig).validate_python(data)


class EmbeddingLLMType(str, Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"


class OpenAIEmbeddingConfig(BaseModel):
    type: Literal[EmbeddingLLMType.OpenAI]
    endpoint: Annotated[str, Field(
        description="The Azure OpenAI endpoint URL",
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


class AzureOpenAIEmbeddingConfig(BaseModel):
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


EmbeddingConfig = Annotated[OpenAIEmbeddingConfig | AzureOpenAIEmbeddingConfig, Field(
    description="Configuration for the embedding model",
    discriminator="type",
)]
