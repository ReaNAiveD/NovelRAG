from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

from novelrag.config.llm import AzureOpenAIEmbeddingConfig, OpenAIEmbeddingConfig
from novelrag.llm.oai.types import AsyncOpenAIClient
from novelrag.llm.types import EmbeddingLLM


class OpenAIEmbeddingLLM(EmbeddingLLM):
    def __init__(self, client: AsyncOpenAIClient, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_config(cls, config: AzureOpenAIEmbeddingConfig | OpenAIEmbeddingConfig):
        if isinstance(config, AzureOpenAIEmbeddingConfig):
            if config.api_key:
                credential = {"api_key": config.api_key}
            else:
                from azure.identity import DefaultAzureCredential
                credential = {
                    "azure_ad_token_provider": get_bearer_token_provider(
                        DefaultAzureCredential(),
                        "https://cognitiveservices.azure.com/.default"
                    )
                }
            client = AsyncAzureOpenAI(
                azure_endpoint=config.endpoint,
                azure_deployment=config.deployment,
                api_version=config.api_version,
                timeout=config.timeout,
                **credential,
            )
        else:
            client = AsyncOpenAI(
                base_url=config.endpoint,
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
