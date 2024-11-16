from typing import Dict, Any

from openai import BaseModel
from pydantic import RootModel, Field
from typing_extensions import Annotated


class AspectContextConfig(BaseModel):
    config: Annotated[Dict[str, Any], Field(description="")]


AspectsConfig = RootModel[Dict[str, AspectContextConfig]]
