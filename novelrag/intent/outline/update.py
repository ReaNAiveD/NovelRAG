from novelrag.intent.intent import Intent, ActionResult
from novelrag.exceptions import InvalidIndexError, InvalidMessageFormatError
from novelrag.core.operation import OperationType
from novelrag.editors.outline.definitions import OutlineActionConfig
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.llm import AzureAIClient
from novelrag.model.outline import Outline

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You are a writer contributing to a collaborative novel project, specializing in refining outline events to strengthen narrative flow and structure.

---Guideline---
Help refine the existing event while considering:
    1. Hierarchy Context:
        - Parent event: {current_path}
        - Event's level: Main plot point / Sub-event / Specific scene
        - Connections to sibling events
    
    2. Narrative Integration:
        - Event's purpose and impact
        - Flow with surrounding events
        - Character development
        - Plot progression at appropriate scope
        - Clear communication of what happens
        - Creation of narrative momentum
    
    3. When suggesting improvements:
        - Identify specific parts that are too strong/weak/unclear/redundant
        - Explain why, considering the event's position and relationships
        - Provide focused modification suggestions
        - Consider impact on both local and broader narrative
    
    4. If no specific direction is given, analyze the event's current position and suggest improvements

---Context---
Current location in outline: {current_path}

Surrounding events:
{events}

Target event to refine:
{target_event}
"""

SUBMIT_PROMPT = """
{language_instruction}

---Role---
You are a professional editor specializing in story outline development. Your task is to synthesize the discussion to refine an existing event while preserving narrative flow.

---Guidelines---
1. Analysis:
   - Review the conversation for key refinements
   - Identify the event's core purpose and improvements
   - Note specific changes requested

2. Event Refinement:
   - Strengthen narrative impact
   - Incorporate discussion feedback
   - Maintain connections to surrounding events
   - Enhance plot progression

---Context---
Current location: {current_path}

Surrounding events:
{events}

Target event:
{target_event}

Discussion history:
{conversation}

---Output Format---
Provide your response in two parts:
1. Brief analysis of key refinements
2. End with "**Updated Event:** {{EVENT}}"
"""

@outline_registry.register('update')
class UpdateIntent(Intent):
    def __init__(self, idx: int, outline: Outline, current_location: EventLocation,
                 oai_config: dict, chat_params: dict):
        super().__init__()
        self.outline = outline
        self.current_location = current_location
        self.idx = idx
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AzureAIClient(**oai_config)
        self.history = []
        self.current_events = self.current_location.get_current_events(outline)

        if idx < 0 or idx >= len(self.current_events):
            raise InvalidIndexError(idx, len(self.current_events) - 1, "outline")

        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction="Use the same language as the user's input.",
            current_path=' / '.join(self.current_location.get_path_names(outline)),
            events=self._format_events_list(),
            target_event=self.current_events[idx]['content']
        )

    def _format_events_list(self) -> str:
        return "\n".join(f"{i}. {event['content']}" 
                        for i, event in enumerate(self.current_events))

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig):
        try:
            split_msg = input_msg.split(maxsplit=1)
            idx = int(split_msg[0])
            message = split_msg[1] if len(split_msg) > 1 else None
            return cls(idx, **config), message
        except ValueError:
            raise InvalidMessageFormatError(
                'update',
                'event',
                input_msg,
                "update INDEX [message]"
            )

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

    async def _summary_update(self, message: str | None) -> str:
        system_message = SUBMIT_PROMPT.format(
            language_instruction="Use the same language as the user's input.",
            conversation="\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history),
            current_path=' / '.join(self.current_location.get_path_names(self.outline)),
            events=self._format_events_list(),
            target_event=self.current_events[self.idx]['content']
        )
        message = message or "Please summarize our discussion and provide the final updated event."
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message}
            ],
            **self.chat_params
        )
        return resp.choices[0].message.content.split('**Updated Event:** ')[-1]

    async def submit(self, message: str | None):
        new_event = message or await self._summary_update(None)
        current_path = self.current_location.path
        update_path = f"{'.'.join(map(str, current_path))}.{self.idx}"
        return ActionResult.operation(OperationType.UPDATE, f'events.{update_path}', new_event)
