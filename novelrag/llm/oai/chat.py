from novelrag.config.llm import AzureOpenAIChatConfig, OpenAIChatConfig, DeepSeekChatConfig
from novelrag.llm.types import AsyncOpenAIClient, ChatLLM
from openai import AsyncAzureOpenAI, AsyncOpenAI


class OpenAIChatLLM(ChatLLM):
    def __init__(
            self,
            client: AsyncOpenAIClient,
            model: str,
            *,
            json_supports: bool = False,
            chat_params: dict | None = None,
    ):
        self.client = client
        self.model = model
        self.json_supports = json_supports
        self.chat_params: dict = chat_params or {}

    @classmethod
    def from_config(cls, config: AzureOpenAIChatConfig | OpenAIChatConfig | DeepSeekChatConfig) -> 'OpenAIChatLLM':
        if isinstance(config, AzureOpenAIChatConfig):
            client = AsyncAzureOpenAI(
                azure_endpoint=config.endpoint,
                azure_deployment=config.deployment,
                api_version=config.api_version,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        else:
            client = AsyncOpenAI(
                base_url=config.endpoint,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        chat_params = config.chat_params()
        model = config.model
        return cls(client, model, json_supports=config.json_supports, chat_params=chat_params)

    async def chat(self, messages: list[dict], **params) -> str:
        """
        Send a chat completion request to Azure OpenAI

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **params: Additional parameters for the chat completion API

        Returns:
            ChatCompletion response
        """
        if not self.json_supports and 'response_format' in params:
            params.pop('response_format')
        resp = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **self.chat_params,
            **params,
        )
        return resp.choices[0].message.content
