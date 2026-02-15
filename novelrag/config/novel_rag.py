import logging
import warnings

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from novelrag.config.handler import HandlerConfig
from novelrag.config.llm import EmbeddingConfig, ChatConfig
from novelrag.config.resource import VectorStoreConfig

_logger = logging.getLogger(__name__)


class NovelRagConfig(BaseModel):
    chat_llm: Annotated[ChatConfig, Field()]
    embedding: Annotated[EmbeddingConfig, Field()]
    vector_store: Annotated[VectorStoreConfig, Field()]
    language: Annotated[str | None, Field(
        default=None,
        description="Content language for the project. Controls prompt template selection, "
                    "resource content language (IDs, URIs, descriptions), intermediate reasoning, "
                    "and aspect schema descriptions. When unset, follows the language of agent_beliefs. "
                    "If beliefs are also empty, defaults to English.",
    )]
    resource_config: Annotated[str, Field(description='Path to the aspect metadata file', default='aspect.yml')]
    agent_beliefs: Annotated[list[str], Field(description='List of beliefs/constraints for the agent', default_factory=list)]
    default_resource_dir: Annotated[str, Field(description="Default directory for resources of new aspect", default='.')]
    undo_path: Annotated[str | None, Field(description="Path to store undo history", default=None)]
    backlog_path: Annotated[str | None, Field(description="Path to store backlog entries", default=None)]

    @model_validator(mode='before')
    @classmethod
    def _migrate_template_lang(cls, data):
        """Backward compatibility: rename template_lang -> language."""
        if isinstance(data, dict) and 'template_lang' in data:
            if 'language' not in data or data['language'] is None:
                data['language'] = data.pop('template_lang')
                _logger.warning(
                    "Config field 'template_lang' is deprecated. "
                    "Please use 'language' instead."
                )
            else:
                data.pop('template_lang')
                _logger.warning(
                    "Both 'template_lang' and 'language' are set in config. "
                    "Using 'language' value. Please remove 'template_lang'."
                )
        return data
