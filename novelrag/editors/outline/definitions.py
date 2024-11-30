from dataclasses import dataclass
from typing import TypedDict, List

from novelrag.editors.outline.navigation import EventLocation
from novelrag.model import Premise
from novelrag.model.outline import Event, Outline


class OutlineActionConfig(TypedDict):
    outline: Outline
    premise: Premise
    current_location: EventLocation
    oai_config: dict
    chat_params: dict

@dataclass
class OutlineDefinition:
    # Language instruction for all prompts
    LANGUAGE_INSTRUCTION = """Note: Please respond in the same language as the user's input. If the user writes in a specific language, provide your entire response in that language."""

    # Common aspects of story structure
    STORY_STRUCTURE = """
        - Event Name and Purpose
        - Main Characters involved
        - Story Arc Components:
            * Introduction
            * Rising Action
            * Climax
            * Falling Action
            * Resolution
        - Sub-events and their connections"""

    # Common requirements for any event
    EVENT_REQUIREMENTS = """
        - Have a clear and descriptive name
        - List main characters involved
        - Include all story arc components
        - Maintain narrative flow
        - Connect logically to parent/child events
        - Support the overall story structure"""

    @staticmethod
    def format_event(event: Event, indent: int = 0) -> str:
        indent_str = "  " * indent
        result = [f"{indent_str}- {event.name}:"]
        indent_str += "  "
        result.append(f"{indent_str}Main Characters: {', '.join(event.mainCharacters)}")
        result.append(f"{indent_str}Introduction: {event.introduction}")
        result.append(f"{indent_str}Rising Action: {event.risingAction}")
        result.append(f"{indent_str}Climax: {event.climax}")
        result.append(f"{indent_str}Falling Action: {event.fallingAction}")
        result.append(f"{indent_str}Resolution: {event.resolution}")
        
        if event.subEvents:
            result.append(f"{indent_str}Sub-events:")
            for sub_event in event.subEvents:
                result.extend(OutlineDefinition.format_event(sub_event, indent + 2))
        
        return "\n".join(result)

    @staticmethod
    def format_events_list(events: List[Event]) -> str:
        return "\n".join([OutlineDefinition.format_event(event) for event in events])
