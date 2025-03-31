from novelrag.intent.intent import Intent, ActionResult
from novelrag.editors.outline.definitions import OutlineActionConfig, OutlineDefinition
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.llm import AzureAIClient
from novelrag.model.outline import Event, Outline

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You are a writing assistant specializing in story outlines. Your role is to:
- Analyze and discuss existing outline events
- Provide insights and suggestions for event development
- Help brainstorm narrative connections
- Answer questions about outline structure and flow

---Context---
Current location in outline: {current_path}

Current events:
{events}

---Guidelines---
1. Maintain awareness of event hierarchy and narrative flow
2. Provide constructive feedback and suggestions
3. Help identify opportunities to strengthen the outline
4. Consider these aspects in discussions:
{story_structure}

---Requirements---
When discussing events, consider:
{event_requirements}

Feel free to discuss, analyze, or provide suggestions about these events.
"""

@outline_registry.register('_default')
class DefaultIntent(Intent):
    def __init__(self, outline: Outline, current_location: EventLocation, oai_config: dict, chat_params: dict):
        super().__init__()
        self.outline = outline
        self.current_location = current_location
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AzureAIClient(**oai_config)
        self.history = []
        self.definition = OutlineDefinition()
        self.current_events = self._current_events()
        
        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction=self.definition.LANGUAGE_INSTRUCTION,
            current_path=' / '.join(self.current_location.get_path_names(outline)),
            events=self._format_events_list(),
            story_structure=self.definition.STORY_STRUCTURE,
            event_requirements=self.definition.EVENT_REQUIREMENTS
        )

    def _format_events_list(self) -> str:
        return "\n".join(f"{i}. {event}"
                        for i, event in enumerate(self.current_events))

    def _current_events(self) -> list[Event]:
        return self.current_location.get_current_events(self.outline)

    @classmethod
    async def create(cls, msg: str | None, **config: OutlineActionConfig):
        return cls(**config), msg

    async def handle(self, message: str | None) -> ActionResult:
        if not message or len(message.strip()) == 0:
            return ActionResult.message(
                "How can I help you with the outline? Feel free to discuss or ask questions about "
                "any events in the current section."
            )
            
        self.history.append({'role': 'user', 'content': message})
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                *self.history
            ],
            **self.chat_params
        )
        self.history.append({
            'role': resp.choices[0].message.role,
            'content': resp.choices[0].message.content
        })
        return ActionResult.message(resp.choices[0].message.content)
