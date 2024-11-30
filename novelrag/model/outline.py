from pydantic import BaseModel, Field
from typing_extensions import Annotated

from novelrag.core.registry import register_model


class Event(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    mainCharacters: Annotated[list[str], Field(min_length=1)]
    introduction: Annotated[str, Field(default='')]
    risingAction: Annotated[str, Field(default='')]
    climax: Annotated[str, Field(default='')]
    fallingAction: Annotated[str, Field(default='')]
    resolution: Annotated[str, Field(default='')]
    subEvents: Annotated[list['Event'], Field(default_factory=lambda: [])]

    def __str__(self):
        parts = [f"{self.name}:\n  characters: {', '.join(self.mainCharacters)}"]
        
        if self.introduction:
            parts.append(f"  introduction: {self.introduction}")
        if self.risingAction:
            parts.append(f"  risingAction: {self.risingAction}")
        if self.climax:
            parts.append(f"  climax: {self.climax}")
        if self.fallingAction:
            parts.append(f"  fallingAction: {self.fallingAction}")
        if self.resolution:
            parts.append(f"  resolution: {self.resolution}")
        if self.subEvents:
            parts.append(f"  subEvents: {', '.join(sub_event.name for sub_event in self.subEvents)}")
        
        return '\n'.join(parts)


@register_model('outline')
class Outline(BaseModel):
    events: Annotated[list[Event], Field(default_factory=lambda: [])]

    def __str__(self):
        return f"Outline:\n  events: {', '.join(event.name for event in self.events)}"
