"""Goal building utilities for translating user requests into goals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.agenturn.interaction import InteractionContext
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm
from novelrag.utils.language import interaction_directive


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
    async def translate(
            self,
            request: str,
            beliefs: list[str],
            interaction_history: InteractionContext | None = None,
    ) -> Goal: ...


class GoalDecider(Protocol):
    """Protocol for autonomous goal generation."""
    async def next_goal(
            self,
            beliefs: list[str],
            interaction_history: InteractionContext | None = None,
    ) -> Goal | None: ...


class GoalTranslation(BaseModel):
    """LLM response containing a translated goal statement."""
    goal: Annotated[str, Field(description="A clear, concise goal statement translated from the user request.")]


class LLMGoalTranslator(GoalTranslator):
    """LLM-based implementation of GoalTranslator."""

    TEMPLATE_NAME = "translate_request_to_goal.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en", language: str | None = None):
        template_env = TemplateEnvironment(package_name="novelrag.agenturn", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME)
        self._goal_llm = chat_llm.with_structured_output(GoalTranslation)
        self._lang_directive = interaction_directive(language=language, is_autonomous=False)

    @trace_llm("goal_translation")
    async def translate(
            self,
            request: str,
            beliefs: list[str],
            interaction_history: InteractionContext | None = None,
    ) -> Goal:
        """Translate a user request into a structured Goal using LLM.

        Args:
            request: The user's request as a string.
            beliefs: Current agent beliefs to inform the translation.
            interaction_history: Optional recent interaction history for context.

        Returns:
            A Goal object representing the translated goal.
        """
        history_text = interaction_history.format_recent(5) if interaction_history else ""
        prompt = self.template.render(
            request=request,
            beliefs=beliefs,
            interaction_history=history_text,
        )
        response = await self._goal_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Translate the user request into a clear, concise goal statement.")
        ])
        assert isinstance(response, GoalTranslation)
        return Goal(
            description=response.goal.strip(),
            source=UserRequestSource(request=request)
        )
