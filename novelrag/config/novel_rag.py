from pydantic import BaseModel, Field
from typing_extensions import Annotated

from novelrag.config.handler import HandlerConfig
from novelrag.config.llm import EmbeddingConfig, ChatConfig
from novelrag.config.resource import VectorStoreConfig


class NovelRagConfig(BaseModel):
    chat_llm: Annotated[ChatConfig, Field()]
    embedding: Annotated[EmbeddingConfig, Field()]
    vector_store: Annotated[VectorStoreConfig, Field()]
    template_lang: Annotated[str | None, Field(default=None)]
    resource_config: Annotated[str, Field(description='Path to the aspect metadata file', default='aspect.yml')]
    agent_beliefs: Annotated[list[str], Field(description='List of beliefs/constraints for the agent', default_factory=list)]
    default_resource_dir: Annotated[str, Field(description="Default directory for resources of new aspect", default='.')]
    undo_path: Annotated[str | None, Field(description="Path to store undo history", default=None)]
