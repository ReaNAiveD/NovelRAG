"""Goal building utilities for translating user requests into goals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.template import TemplateEnvironment


@dataclass(kw_only=True)
class GoalSource(ABC):
    """Base class for goal origins. Enables tracing where a goal came from."""
    created_at: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass(kw_only=True)
class UserRequestSource(GoalSource):
    """Goal originated from a user request."""
    request: str

    def __str__(self) -> str:
        return f"UserRequest({self.request})"


@dataclass(kw_only=True)
class AutonomousSource(GoalSource):
    """Goal generated autonomously by a decider."""
    decider_name: str
    context: str | None = None

    def __str__(self) -> str:
        if self.context:
            return f"Autonomous[{self.decider_name}]({self.context})"
        return f"Autonomous[{self.decider_name}]"


@dataclass
class Goal:
    """Represents a clear and concise goal for the agent."""
    description: str = field()
    source: GoalSource = field()

    def __str__(self) -> str:
        return f"Goal: {self.description}\nSource Request: {self.source}"


class GoalTranslator(Protocol):
    """Protocol for translating user requests into goals."""
    async def translate(self, request: str, beliefs: list[str]) -> Goal: ...


class GoalDecider(Protocol):
    """Protocol for autonomous goal generation."""
    async def next_goal(self, beliefs: list[str]) -> Goal | None: ...


class LLMGoalTranslator(GoalTranslator):
    """LLM-based implementation of GoalTranslator."""

    TEMPLATE_NAME = "translate_request_to_goal.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en"):
        template_env = TemplateEnvironment(package_name="novelrag.agenturn", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME)
        self.chat_llm = chat_llm

    async def translate(self, request: str, beliefs: list[str]) -> Goal:
        """Translate a user request into a structured Goal using LLM.

        Args:
            request: The user's request as a string.
            beliefs: Current agent beliefs to inform the translation.

        Returns:
            A Goal object representing the translated goal.
        """
        prompt = self.template.render(
            user_request=request,
            beliefs=beliefs
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Translate the user request into a clear, concise goal statement.")
        ])
        assert isinstance(response.content, str), "Expected string response from LLM"
        goal_description = self._extract_goal(response.content)
        return Goal(
            description=goal_description,
            source=UserRequestSource(request=request)
        )

    @staticmethod
    def _extract_goal(response: str) -> str:
        """Extract the goal statement from LLM response."""
        # Look for "**Goal**:" or "Goal:" pattern
        lines = response.strip().split("\n")
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("**goal**:"):
                return line_stripped[9:].strip()
            if line_stripped.lower().startswith("goal:"):
                return line_stripped[5:].strip()
        return response.strip()
