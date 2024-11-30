import json

import pydantic

from novelrag.core.action import Action, ActionResult
from novelrag.core.exceptions import InvalidMessageFormatError, InvalidLLMResponseFormatError
from novelrag.core.operation import OperationType
from novelrag.editors.outline.definitions import OutlineActionConfig, OutlineDefinition
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.llm import AzureAIClient
from novelrag.model import Premise
from novelrag.model.outline import Event, Outline

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You are a story structure expert specializing in breaking down narrative events into meaningful sub-events.

---Task---
Break down the following event into 3-5 sub-events that naturally compose the original event:

{event_details}

---Guidelines---
1. Each sub-event must:
   - Have a clear connection to the parent event
   - Follow a complete narrative arc
   - Involve relevant characters from the parent event
   - Together, fully cover the parent event's content

2. Ensure the sub-events:
   - Flow logically from one to another
   - Maintain narrative continuity
   - Preserve the original story's premises

3. For root node or empty events:
   - If this is the root node (an empty event), create 3-5 major story acts based on the premises
   - Ensure these events align with the overall story premises and maintain narrative coherence

---Story Premises---
{premises_text}

---Output Format---
Return a JSON object with a single "events" array. Each event object must have the following structure:
{{
    "events": [
        {{
            "name": "Brief, descriptive event name",
            "mainCharacters": ["Character 1", "Character 2"],
            "introduction": "Setup and context of the event",
            "risingAction": "Building tension and complications",
            "climax": "Peak moment or critical decision",
            "fallingAction": "Immediate aftermath",
            "resolution": "Final outcome and consequences"
        }}
    ]
}}

Requirements:
- All fields are required strings (except mainCharacters which is an array of strings)
- Provide 3-5 event objects in the array
- Each field should be concise but descriptive
- Do not include any text outside the JSON structure
"""

@outline_registry.register('break')
class BreakAction(Action):
    def __init__(self, outline: Outline, premise: Premise, current_location: EventLocation,
                 oai_config: dict, chat_params: dict):
        super().__init__()
        self.outline = outline
        self.premise = premise
        self.current_location = current_location
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AzureAIClient(**oai_config)
        
        # Get the current event
        self.current_event = self.current_location.get_current_event(outline)
        
        # Check if event has children
        if (isinstance(self.current_event, Event) and self.current_event.subEvents) \
                or (isinstance(self.current_event, Outline) and self.current_event.events):
            raise InvalidMessageFormatError('break', 'outline',
                "This action can only be used on events without sub-events.", None)

        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction=OutlineDefinition.LANGUAGE_INSTRUCTION,
            premises_text='\n'.join(f"  - {premise}" for premise in premise.premises),
            event_details=str(self.current_event)
        )

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig):
        return cls(**config), input_msg

    def _parse_events_data(self, content: str) -> list[Event]:
        """Parse events data from string content into list of Event objects."""
        try:
            events_data = json.loads(content)["events"]
            if not isinstance(events_data, list):
                raise InvalidLLMResponseFormatError(
                    'break', 'outline', content,
                    "Expected JSON array of events"
                )
            return [Event(**event_data) for event_data in events_data]
        except (json.JSONDecodeError, pydantic.ValidationError) as e:
            raise InvalidLLMResponseFormatError(
                'break', 'outline', content,
                f"Invalid event format: {str(e)}"
            )

    async def handle(self, message: str | None) -> ActionResult:
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': message or "Please break down this event into sub-events."}
            ],
            response_format={"type": "json_object"},
            **self.chat_params
        )
        
        sub_events = self._parse_events_data(resp.choices[0].message.content)
        current_path = self.current_location.path
        update_path = ''.join([f'.{seg}.subEvents' for seg in current_path])
        
        try:
            return ActionResult.operation(
                OperationType.PUT,
                f'events{update_path}',
                [event.model_dump() for event in sub_events]
            )
        except InvalidMessageFormatError as e:
            return ActionResult.message(str(e))
