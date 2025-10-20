from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class AspectConfig(BaseModel):
    path: Annotated[str, Field(description='Path to the aspect data file')]
    description: Annotated[str | None, Field(default=None, description='Description of the aspect')]
    children_keys: Annotated[list[str], Field(description='The keys of fields that hold children in resource.', default_factory=lambda: [])]

    model_config = ConfigDict(extra='allow')


class VectorStoreConfig(BaseModel):
    lancedb_uri: Annotated[str, Field()]
    table_name: Annotated[str, Field()]
    overwrite: Annotated[bool, Field(description='Whether the vector table collected from YAML would overwrite the database.', default=True)]
    cleanup_invalid_on_init: Annotated[bool, Field(description='Whether to clean up invalid vectors during repository initialization.', default=True)]
