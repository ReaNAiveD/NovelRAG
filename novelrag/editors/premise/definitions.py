from dataclasses import dataclass
from typing import TypedDict

class PremiseActionConfig(TypedDict):
    premises: list[str]
    oai_config: dict
    chat_params: dict

@dataclass
class PremiseDefinition:
    # Language instruction for all prompts
    LANGUAGE_INSTRUCTION = """Note: Please respond in the same language as the user's input. If the user writes in a specific language, provide your entire response in that language."""

    # Common aspects that can be focused on in a premise
    ASPECT_TYPES = """
        - Main plot arc or core conflict
        - Subplot development or unique plot twist
        - Character backstory and goals
        - World-building elements and setting
        - Thematic exploration and emotional connections
        - Historical or social context
        - Stakes and consequences"""

    # Common requirements for any premise
    PREMISE_REQUIREMENTS = """
        - Be limited to one paragraph
        - Not exceed 200 words
        - Focus on a single, distinct aspect
        - Complement existing premises without redundancy
        - Maintain clear boundaries with other premises
        - Create opportunities for narrative integration"""

    # Common integration guidelines
    INTEGRATION_GUIDELINES = """
        - Fill gaps in the overall narrative
        - Add new dimensions without contradicting existing premises
        - Maintain clear boundaries between your focus and other premises
        - Create opportunities for interaction without overlap"""

    @staticmethod
    def format_premises_list(premises: list[str]) -> str:
        return '\n'.join([f'  {idx}. {p}' for idx, p in enumerate(premises)])

    @staticmethod
    def format_conversation(history: list[dict]) -> str:
        return '\n\n'.join([f'{chat["role"]}:\n{chat["content"]}' for chat in history])

    @staticmethod
    def get_default_submit_message() -> str:
        return 'Please summary and update the premise.\nThink step by step. Provide the final premise in the last paragraph in the format of "**Updated Premise:** {PREMISE}".'
