from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ConversationItem(BaseModel):
    role: Annotated[str, Field()]
    aspect: Annotated[str | None, Field(default=None)]
    intent: Annotated[str | None, Field(default=None)]
    message: Annotated[str, Field()]
