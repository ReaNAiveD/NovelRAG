from novelrag.core.action import Action, ActionResult
from novelrag.core.exceptions import InvalidIndexError, InvalidMessageFormatError
from novelrag.core.operation import OperationType
from novelrag.editors.premise.definitions import PremiseDefinition, PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry
from novelrag.llm import AzureAIClient

SYSTEM_PROMPT = """
{language_instruction}

---Role---

You are a writer contributing to a collaborative novel project, specializing in refining existing premises to strengthen their distinct aspects.

---Guideline---

Help refine the existing premise while maintaining its unique aspect and role in the narrative. Consider:
    1. Aspect Focus: Strengthen the premise's focus on its specific aspect:
{aspect_types}
    2. Clarity: Ensure the premise clearly communicates its unique aspect
    3. Integration: Maintain and enhance how the premise:
        - Fills its specific role in the narrative
        - Complements other premises without overlap
        - Maintains clear boundaries with other premises
        - Creates opportunities for interaction
    4. If no specific direction is given, analyze the premise's current focus and suggest improvements

When suggesting improvements, identify specific parts that need refinement:
- Point out which parts are too strong/weak/unclear/redundant
- Explain why these parts need adjustment
- Provide specific suggestions for modifications
- Consider how changes affect the premise's role in the larger narrative

---Limitation---

Your refined premise should be limited to one paragraph and no more than 200 words.

---Context---

The following premises each cover different aspects of our collaborative novel:
{premises}

---Goal---

Refine the target premise while:
1. Strengthening its focus on its distinct aspect
2. Maintaining clear boundaries with other premises
3. Enhancing its contribution to the larger narrative
4. Preserving its unique role in the story structure

Target Premise:
{target_premise}
"""

SUBMIT_PROMPT = """
{language_instruction}

---Role---

You are a professional editor specializing in novel premise development. Your task is to synthesize the discussion between the writer and AI assistant to refine an existing premise while preserving its distinct role in the narrative.

---Guidelines---

1. Analysis:
   - Review the conversation for key refinements discussed
   - Identify the premise's core aspect and how it's being enhanced
   - Note specific improvements requested by the writer

2. Premise Refinement:
   - Strengthen the premise's focus on its distinct aspect
   - Incorporate discussion feedback and improvements
   - Maintain its unique role while avoiding overlap with other premises
   - Enhance its contribution to the overall narrative

---Requirements---

The refined premise must:
{requirements}

---Context---

Existing premises:
{premises}

Target premise to refine:
{target_premise}

Discussion history:
{conversation}

---Output Format---

Provide your response in two parts:
1. Brief analysis of key refinements from the discussion
2. End with "**Updated Premise:** {{PREMISE}}"
"""

@premise_registry.register('update')
class UpdateAction(Action):
    def __init__(self, idx: int, premises: list[str], oai_config: dict, chat_params: dict):
        super().__init__()
        self.premises = premises
        self.idx = idx
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.oai_client = AzureAIClient(**oai_config)
        self.history = []
        self.definition = PremiseDefinition()
        self.system_prompt = SYSTEM_PROMPT.format(
            language_instruction=self.definition.LANGUAGE_INSTRUCTION,
            premises=self.definition.format_premises_list(self.premises),
            target_premise=self.premises[idx],
            aspect_types=self.definition.ASPECT_TYPES
        )

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        try:
            split_msg = input_msg.split(maxsplit=1)
            idx = int(split_msg[0])
            if idx < 0 or idx >= len(config['premises']):
                raise InvalidIndexError(idx, len(config['premises']) - 1, "premise")
            message = split_msg[1] if len(split_msg) > 1 else None
            return cls(idx, **config), message
        except ValueError:
            raise InvalidMessageFormatError(
                'update', 
                'premise', 
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
            language_instruction=self.definition.LANGUAGE_INSTRUCTION,
            conversation=self.definition.format_conversation(self.history),
            premises=self.definition.format_premises_list(self.premises),
            target_premise=self.premises[self.idx],
            requirements=self.definition.PREMISE_REQUIREMENTS
        )
        message = message or self.definition.get_default_submit_message()
        resp = await self.oai_client.chat(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message}
            ],
            **self.chat_params
        )
        return resp.choices[0].message.content.split('**Updated Premise:** ')[-1]

    async def submit(self, message: str | None):
        new_premise = message or await self._summary_update(None)
        return ActionResult.operation(OperationType.PUT, f'premises.{self.idx}', new_premise)
