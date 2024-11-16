from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

class AzureAIClient:
    def __init__(self, **config):
        self.client = AsyncAzureOpenAI(**config)
        
    async def chat(self, messages: list[dict], **params) -> ChatCompletion:
        """
        Send a chat completion request to Azure OpenAI
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **params: Additional parameters for the chat completion API
            
        Returns:
            ChatCompletion response
        """
        return await self.client.chat.completions.create(
            messages=messages,
            **params
        )
