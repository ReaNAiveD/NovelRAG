from openai import AsyncAzureOpenAI

from novelrag.action import Action, ActionResult
from novelrag.aspect_editors.premise.definitions import PremiseDefinition, PremiseActionConfig

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You are a helpful writing assistant discussing story premises. Your role is to:
- Analyze and discuss existing premises
- Provide insights and suggestions
- Help brainstorm ideas
- Answer questions about premise development

---Context---
Current premises in the story:
{premises}

---Guidelines---
1. Maintain awareness of existing premises while discussing
2. Provide constructive feedback and suggestions
3. Help identify gaps or opportunities in the narrative
4. Consider these aspects in discussions:
{aspect_types}

Feel free to discuss, analyze, or provide suggestions about these premises.
"""

class DefaultAction(Action):
    def __init__(self, premises: list[str], oai_config: dict, chat_params: dict):
        super().__init__()
        self.premises = premises
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AsyncAzureOpenAI(**oai_config)
        self.history = []
        self.definition = PremiseDefinition()
        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction=self.definition.LANGUAGE_INSTRUCTION,
            premises=self.definition.format_premises_list(self.premises),
            aspect_types=self.definition.ASPECT_TYPES
        )

    @property
    def name(self):
        return 'default'

    @classmethod
    async def create(cls, msg: str | None, **config: PremiseActionConfig):
        return cls(**config), msg

    async def handle(self, message: str | None) -> ActionResult:
        if not message:
            return ActionResult.message("How can I help you with the premises? Feel free to discuss or ask questions about any aspect of the story.")
            
        self.history.append({'role': 'user', 'content': message})
        resp = await self.oai_client.chat.completions.create(
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