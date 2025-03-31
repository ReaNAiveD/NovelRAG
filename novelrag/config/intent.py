from pydantic import BaseModel, Field
from typing_extensions import Annotated, Any


class IntentConfig(BaseModel):
    name: Annotated[str | None, Field(default=None, description='Override the dict key')]
    cls: Annotated[str, Field(description='Class path of the intent')]
    kwargs: Annotated[dict[str, Any], Field(default_factory=lambda: {}, description='kwargs to initialize the class')]
