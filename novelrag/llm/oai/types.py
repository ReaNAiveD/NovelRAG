from openai import AsyncAzureOpenAI, AsyncOpenAI

# Type alias for OpenAI clients
AsyncOpenAIClient = AsyncAzureOpenAI | AsyncOpenAI
