from abc import ABC, abstractmethod

from openai import AsyncAzureOpenAI, AsyncOpenAI

# Type alias for OpenAI clients
AsyncOpenAIClient = AsyncAzureOpenAI | AsyncOpenAI


class ChatLLM(ABC):
    """Abstract base class for chat language models."""
    
    @abstractmethod
    async def chat(self, messages: list[dict], **params) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **params: Additional parameters for the chat completion API

        Returns:
            Generated response text
        """
        pass


class EmbeddingLLM(ABC):
    """Abstract base class for embedding language models."""
    
    @abstractmethod
    async def embedding(self, message: str, **params) -> list[list[float]]:
        """Generate embeddings for input text.

        Args:
            message: Input text to embed
            **params: Additional parameters for the embedding API

        Returns:
            List of embedding vectors
        """
        pass
