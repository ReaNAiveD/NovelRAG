from azure.ai.inference.aio import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from novelrag.config.llm import AzureOpenAIEmbeddingConfig
from novelrag.llm import EmbeddingLLM


class AzureAIEmbeddingLLM(EmbeddingLLM):
    def __init__(
            self,
            client: EmbeddingsClient,
            model: str,):
        self.client = client
        self.model = model

    @classmethod
    def from_config(cls, config: AzureOpenAIEmbeddingConfig):
        if isinstance(config, AzureOpenAIEmbeddingConfig):
            endpoint = config.endpoint.rstrip('/') + f'/openai/deployments/{config.deployment}'
            if config.api_key:
                credential = {"credential": AzureKeyCredential(config.api_key)}
            else:
                from azure.identity import DefaultAzureCredential
                credential = {
                    "credential": DefaultAzureCredential(),
                    "credential_scopes": ["https://cognitiveservices.azure.com/.default"],
                }
            client = EmbeddingsClient(
                endpoint=endpoint,
                api_version=config.api_version,
                model=config.model,
                **credential
            )
        else:
            client = EmbeddingsClient(
                endpoint=config.endpoint,
                credential=AzureKeyCredential(config.api_key),
                api_version=config.api_version,
                model=config.model,
            )
        model = config.model
        return cls(client, model)

    async def embedding(self, message: str, **params) -> list[list[float]]:
        resp = await self.client.embed(
            input=message,
            encoding_format="float",
            **params)
        return [d.embedding for d in resp.data]
