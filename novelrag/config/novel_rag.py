from pydantic import BaseModel, Field
from typing_extensions import Annotated

from novelrag.config.intent import IntentConfig
from novelrag.config.llm import EmbeddingConfig, ChatConfig
from novelrag.config.resource import VectorStoreConfig


class NovelRagConfig(BaseModel):
    embedding: Annotated[EmbeddingConfig, Field()]
    chat_llm: Annotated[ChatConfig | None, Field(default=None)]
    vector_store: Annotated[VectorStoreConfig, Field()]
    template_lang: Annotated[str | None, Field(default=None)]
    resource_config: Annotated[str, Field(description='Path to the aspect metadata file', default='aspect.yml')]
    scopes: Annotated[dict[str, Annotated[dict[str, IntentConfig], Field(default_factory=dict)]], Field(default_factory=dict)]
    intents: Annotated[dict[str, IntentConfig], Field(description="Intents for session level")]
    default_resource_dir: Annotated[str, Field(description="Default directory for resources of new aspect", default='.')]
