from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from novelrag.config.llm import AzureOpenAIChatConfig, OpenAIChatConfig, DeepSeekChatConfig
from novelrag.llm import ChatLLM


class AzureAIChatLLM(ChatLLM):
    def __init__(
            self,
            client: ChatCompletionsClient,
            model: str,
            *,
            json_supports: bool = False,
            chat_params: dict | None = None,):
        self.client = client
        self.model = model
        self.json_supports = json_supports
        self.chat_params: dict = chat_params or {}

    @classmethod
    def from_config(cls, config: AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig) -> 'AzureAIChatLLM':
        chat_params = {
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'n': config.n,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty,
        }
        if isinstance(config, AzureOpenAIChatConfig):
            if config.api_key:
                credential = {"credential": AzureKeyCredential(config.api_key)}
            else:
                from azure.identity import DefaultAzureCredential
                credential = {
                    "credential": DefaultAzureCredential(),
                    "credential_scopes": ["https://cognitiveservices.azure.com/.default"],
                }
            endpoint = config.endpoint.rstrip('/') + f'/openai/deployments/{config.deployment}'
            client = ChatCompletionsClient(
                endpoint=endpoint,
                api_version=config.api_version,
                model=config.model,
                **chat_params,
                **credential
            )
        else:
            client = ChatCompletionsClient(
                endpoint=config.endpoint,
                credential=AzureKeyCredential(config.api_key),
                model=config.model,
                **chat_params
            )
        return cls(client, config.model, json_supports=config.json_supports)

    async def chat(self, messages: list[dict], **params) -> str:
        if not self.json_supports and 'response_format' in params:
            params.pop('response_format')
        if 'max_completion_tokens' in params:
            params['max_tokens'] = params.pop('max_completion_tokens')
        resp = await self.client.complete(
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content
