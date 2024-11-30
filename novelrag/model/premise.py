from pydantic import BaseModel, Field
from typing_extensions import Annotated

from novelrag.core.registry import register_model


@register_model('premise')
class Premise(BaseModel):
    premises: Annotated[list[str], Field(default_factory=lambda: [])]
