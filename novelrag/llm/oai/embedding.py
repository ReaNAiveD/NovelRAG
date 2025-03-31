from openai import AsyncAzureOpenAI

from novelrag.config.llm import AzureOpenAIEmbeddingConfig
from novelrag.llm.types import AsyncOpenAIClient, EmbeddingLLM


class OpenAIEmbeddingLLM(EmbeddingLLM):
    def __init__(self, client: AsyncOpenAIClient, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_config(cls, config: AzureOpenAIEmbeddingConfig):
        client = AsyncAzureOpenAI(
            azure_endpoint=config.endpoint,
            azure_deployment=config.deployment,
            api_version=config.api_version,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        model = config.model
        return cls(client, model)

    async def embedding(self, message: str, **params) -> list[list[float]]:
        # By default, the length of the embedding vector will be 1536 for text-embedding-3-small
        # or 3072 for text-embedding-3-large
        embedding = await self.client.embeddings.create(
            input=message,
            model=self.model,
            encoding_format="float",
            # dimensions=3072,
            **params
        )
        return [d.embedding for d in embedding.data]
