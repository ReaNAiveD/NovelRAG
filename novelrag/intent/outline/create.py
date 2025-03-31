import json

import pydantic

from novelrag.intent.intent import Intent, ActionResult
from novelrag.exceptions import InvalidMessageFormatError, InvalidLLMResponseFormatError
from novelrag.core.operation import OperationType
from novelrag.editors.outline.definitions import OutlineActionConfig
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.llm import AzureAIClient
from novelrag.model import Premise
from novelrag.model.outline import Outline, Event

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You are a writer contributing to a collaborative novel project, specializing in creating new outline events that enhance narrative flow and structure.

---Story Premises and Guidelines---
The following premises must be strictly followed when creating new events:
{premises_text}

---Event Structure---
Each event in the outline contains the following elements:
1. Name: A clear, descriptive title for the event
2. Main Characters: Key characters involved in this event
3. Narrative Flow:
   - Introduction: Setting the scene and initial situation
   - Rising Action: Building tension and complications
   - Climax: The peak moment or turning point
   - Falling Action: The immediate aftermath
   - Resolution: How the event concludes and its impact

---Guidelines---
When discussing the new event, address each element explicitly:
1. Hierarchy Context:
   - Parent event: {current_path}
   - Event's level: Main plot point / Sub-event / Specific scene
   - Connections to existing events

2. Narrative Development:
   - Suggest a clear, descriptive name for the event
   - Identify which main characters should be involved
   - Describe how the scene unfolds through each narrative stage:
     * How does it begin? (Introduction)
     * What complications arise? (Rising Action)
     * What is the key moment? (Climax)
     * What immediate effects follow? (Falling Action)
     * How does it conclude? (Resolution)

3. Story Integration:
   - Explain how this event connects to surrounding events
   - Discuss its impact on character development
   - Show how it advances the overall plot

---Context---
Current location in outline: {current_path}

Surrounding events:
{events}

If there are no surrounding events, this is the root node of the event tree. You should create a root event that covers the entire story.

Please discuss these elements naturally in conversation, helping the user develop each aspect of the event.
"""

SUBMIT_PROMPT = """
{language_instruction}

---Role---
You are a professional editor specializing in story outline development. Your task is to synthesize the discussion into a structured event format.

---Story Premises and Guidelines---
The following premises must be strictly followed when creating new events:
{premises_text}

---Guidelines---
Review the discussion and create a complete event that includes:

1. Event Identification:
   - Create a clear, descriptive name that captures the event's essence
   - List all main characters who play significant roles

2. Narrative Flow Analysis:
   - Introduction: Capture the established setting and initial situation
   - Rising Action: Include the discussed complications and tension build-up
   - Climax: Describe the agreed-upon turning point or peak moment
   - Falling Action: Detail the immediate consequences
   - Resolution: Summarize the conclusion and lasting impact

3. Integration Check:
   - Ensure alignment with parent event: {current_path}
   - Verify connections to surrounding events
   - Confirm it advances the overall narrative

---Context---
Current location: {current_path}

Surrounding events:
{events}

Discussion history:
{conversation}

---Output Format---
Provide your response as a JSON object with exactly these fields:
{{
    "name": "Event Name",
    "mainCharacters": ["Character 1", "Character 2"],
    "introduction": "Introduction text",
    "risingAction": "Rising action text",
    "climax": "Climax text",
    "fallingAction": "Falling action text",
    "resolution": "Resolution text"
}}

Note: All fields are required. Ensure each narrative section (introduction through resolution) contains meaningful content that reflects the discussion.
"""

@outline_registry.register('create')
class CreateIntent(Intent):
    def __init__(self, outline: Outline, premise: Premise, current_location: EventLocation,
                 oai_config: dict, chat_params: dict):
        super().__init__()
        self.outline = outline
        self.premise = premise
        self.current_location = current_location
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AzureAIClient(**oai_config)
        self.history = []
        self.current_events = self.current_location.get_current_events(outline)
        
        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction="Use the same language as the user's input.",
            premises_text='\n'.join(f"  - {premise}" for premise in premise.premises),
            current_path=' / '.join(self.current_location.get_path_names(outline)),
            events=self._format_events_list()
        )

    def _format_events_list(self) -> str:
        return "\n".join(f"{i}. {event['content']}" 
                        for i, event in enumerate(self.current_events))

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig):
        return cls(**config), input_msg

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            self.history.append({'role': 'user', 'content': message})
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                *self.history
            ],
            **self.chat_params,
        )
        self.history.append({
            'role': resp.choices[0].message.role,
            'content': resp.choices[0].message.content
        })
        return ActionResult.message(resp.choices[0].message.content)

    async def handle_command(self, command: str, message: str | None):
        match command:
            case 'submit' | 's' | 'yes' | 'y':
                return await self.submit(message)
        return await super().handle_command(command, message)

    def _parse_event_data(self, content: str) -> Event:
        """Parse event data from string content into Event object."""
        try:
            # Try parsing as JSON first
            event_data = json.loads(content)
            return Event(**event_data)
        except (json.JSONDecodeError, pydantic.ValidationError):
            # If JSON parsing fails, raise an error
            raise InvalidLLMResponseFormatError(
                'create', 'outline', content,
                "JSON object with required fields: name and mainCharacters"
            )

    async def _summary_create(self, message: str | None) -> Event:
        system_message = SUBMIT_PROMPT.format(
            language_instruction="Use the same language as the user's input.",
            premises_text='\n'.join(f"  - {premise}" for premise in self.premise.premises),
            conversation="\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history),
            current_path=' / '.join(self.current_location.get_path_names(self.outline)),
            events=self._format_events_list()
        )
        message = message or "Please summarize our discussion and provide the final new event."
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message}
            ],
            **self.chat_params
        )
        response_content = resp.choices[0].message.content
        return self._parse_event_data(response_content)

    async def submit(self, message: str | None):
        event = await self._summary_create(message)
        current_path = self.current_location.path
        new_idx = len(self.current_events)
        update_path = f"{'.subEvents.'.join(map(str, current_path))}.subEvents.{new_idx}"
        try:
            return ActionResult.operation(
                OperationType.NEW,
                f'events.{update_path}',
                event.model_dump()
            )
        except InvalidMessageFormatError as e:
            return ActionResult.message(str(e))
