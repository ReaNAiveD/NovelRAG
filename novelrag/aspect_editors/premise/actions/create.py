from openai import AsyncAzureOpenAI

from novelrag.action import Action, ActionResult
from novelrag.operation import OperationType
from novelrag.aspect_editors.premise.definitions import PremiseDefinition, PremiseActionConfig

SYSTEM_PROMPT = """
{language_instruction}

---Role---

You are a writer contributing to a collaborative novel project, where each premise focuses on a distinct aspect of the story.

---Guideline---

Craft a novel premise that addresses a unique aspect not covered by existing premises while contributing to the overall narrative. Consider:
    1. Distinct Focus: Choose an unexplored aspect of the novel such as:
{aspect_types}
    2. Interest: Ensure your chosen aspect captures attention while avoiding overlap with existing premises.
    3. Depth: Develop your specific aspect thoroughly, complementing but not repeating themes or elements from other premises.
    4. Complementary Integration:
{integration_guidelines}

---Limitation---

Your premise should be limited to one paragraph and no more than 200 words.

---Context---

The following premises each cover different aspects of our collaborative novel:
{premises}

---Goal---

Create an original premise that:
1. Focuses on a distinct, unexplored aspect of the story
2. Avoids redundancy with existing premises
3. Contributes to the larger narrative while maintaining its unique focus
4. Fills a gap in the overall story structure
"""

SUBMIT_PROMPT = """
{language_instruction}

---Role---

You are a professional editor specializing in novel premise development. Your task is to synthesize the discussion between the writer and AI assistant to create a refined, focused premise that maintains the collaborative novel's integrity.

---Guidelines---

1. Analysis:
   - Review the conversation for key story elements and specific aspects discussed
   - Identify the unique aspect being developed (plot, character, world-building, etc.)
   - Note any specific requirements or refinements requested by the writer

2. Premise Refinement:
   - Ensure the premise maintains focus on its distinct aspect
   - Incorporate feedback and improvements from the discussion
   - Verify it doesn't overlap with existing premises
   - Confirm it adds value to the overall narrative

---Requirements---

The final premise must:
{requirements}

---Context---

Existing premises:
{premises}

Discussion history:
{conversation}

---Output Format---

Provide your response in two parts:
1. Brief analysis of key points from the discussion
2. End with "**Updated Premise:** {{PREMISE}}"
"""

class CreateAction(Action):
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
            aspect_types=self.definition.ASPECT_TYPES,
            integration_guidelines=self.definition.INTEGRATION_GUIDELINES
        )

    @property
    def name(self):
        return 'create'

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        return cls(**config), input_msg

    async def handle(self, message: str | None) -> ActionResult:
        self.history.append({'role': 'user', 'content': message or 'Please give some suggestions.'})
        resp = await self.oai_client.chat.completions.create(
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

    async def _summary_create(self, message: str | None) -> str:
        system_message = SUBMIT_PROMPT.format(
            language_instruction=self.definition.LANGUAGE_INSTRUCTION,
            conversation=self.definition.format_conversation(self.history),
            premises=self.definition.format_premises_list(self.premises),
            requirements=self.definition.PREMISE_REQUIREMENTS
        )
        message = message or self.definition.get_default_submit_message()
        resp = await self.oai_client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message}
            ],
            **self.chat_params
        )
        return resp.choices[0].message.content.split('**Updated Premise:** ')[-1]

    async def submit(self, message: str | None):
        new_premise = message or await self._summary_create(None)
        return ActionResult.operation(OperationType.NEW, f'premises.{len(self.premises)}', new_premise)
