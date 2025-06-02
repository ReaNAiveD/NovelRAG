from pydantic import BaseModel, Field
from typing_extensions import Annotated

from novelrag.config.intent import IntentConfig
from novelrag.config.llm import EmbeddingConfig, ChatConfig
from novelrag.config.resource import VectorStoreConfig


class AspectConfig(BaseModel):
    path: Annotated[str | None, Field(description='Path to the aspect data file', default=None)]
    children_keys: Annotated[list[str], Field(description='The keys of fields that hold children in resource.', default_factory=lambda: [])]
    intents: Annotated[dict[str, IntentConfig], Field()]


class NovelRagConfig(BaseModel):
    embedding: Annotated[EmbeddingConfig, Field()]
    chat_llm: Annotated[ChatConfig | None, Field(default=None)]
    vector_store: Annotated[VectorStoreConfig, Field()]
    template_lang: Annotated[str | None, Field(default=None)]
    aspects: Annotated[dict[str, AspectConfig], Field()]
    intents: Annotated[dict[str, IntentConfig], Field(description="Intents for session level")]
