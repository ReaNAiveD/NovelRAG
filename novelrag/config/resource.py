from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ResourceConfig(BaseModel):
    path: Annotated[str, Field(description='Path to the aspect data file')]
    children_keys: Annotated[list[str], Field(description='The keys of fields that hold children in resource.', default_factory=lambda: [])]


class VectorStoreConfig(BaseModel):
    lancedb_uri: Annotated[str, Field()]
    table_name: Annotated[str, Field()]
    overwrite: Annotated[bool, Field(description='Whether the vector table collected from YAML would overwrite the database.', default=True)]
