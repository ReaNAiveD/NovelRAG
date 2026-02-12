from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import JsonSchemaFormat
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
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty,
        }
        if isinstance(config, AzureOpenAIChatConfig):
            if config.api_key:
                credential = {"credential": AzureKeyCredential(config.api_key)}
            else:
                from azure.identity.aio import DefaultAzureCredential
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
        return cls(client, config.model, json_supports=config.json_supports, chat_params=chat_params)

    async def chat(self, messages: list[dict], **params) -> str:
        # Convert response_format if provided and json_supports is True
        if self.json_supports and 'response_format' in params:
            response_format = params['response_format']
            if isinstance(response_format, dict) and response_format.get('type') == 'json_schema':
                # Convert OpenAI format to Azure AI Inference format
                json_schema_data = response_format.get('json_schema', {})
                params['response_format'] = JsonSchemaFormat(
                    name=json_schema_data.get('name', 'structured_response'),
                    schema=json_schema_data.get('schema', {}),
                    description=json_schema_data.get('description'),
                    strict=json_schema_data.get('strict', True)
                )
            elif response_format == 'json_object':
                # Keep simple string format as-is
                params['response_format'] = response_format
        elif not self.json_supports and 'response_format' in params:
            params.pop('response_format')
            
        # Merge chat_params with params
        merged_params = {**self.chat_params, **params}
        
        if 'max_completion_tokens' in merged_params:
            merged_params['max_tokens'] = merged_params.pop('max_completion_tokens')

        resp = await self.client.complete(
            messages=messages,
            model=self.model,
            stream=False,
            **merged_params,
        )
        return resp.choices[0].message.content
