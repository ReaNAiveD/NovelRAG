from typing import Dict, Any

from openai import BaseModel
from pydantic import RootModel, Field
from typing_extensions import Annotated


AspectContextConfig= Dict[str, Any]

AspectsConfig = Dict[str, AspectContextConfig]


class OaiConfig(BaseModel):
    azure_endpoint: str
    azure_deployment: str
    api_version: str
    api_key: str
    timeout: int


class ChatParams(BaseModel):
    model: str
    temperature: float
    top_p: float


class StorageConfig(BaseModel):
    model: str
    file_path: str


NovelStorageConfig = Dict[str, StorageConfig]


class NovelRagConfig(BaseModel):
    storage: NovelStorageConfig
    oai_config: OaiConfig
    chat_params: ChatParams
    aspects: AspectsConfig
