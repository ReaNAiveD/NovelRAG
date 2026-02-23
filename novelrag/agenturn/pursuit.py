"""Data structures for goal pursuit and execution tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.procedure import ExecutionContext
from novelrag.agenturn.tool import SchematicTool
from novelrag.agenturn.interaction import InteractionContext
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from .step import OperationOutcome, OperationPlan, Resolution

logger = logging.getLogger(__name__)


class PursuitStatus(Enum):
    """Status of a goal pursuit."""
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class PursuitOutcome:
    """Represents the final outcome of a goal pursuit."""
    goal: Goal
    reason: str  # Why goal pursuit ended
    response: str  # User-facing completion message
    status: PursuitStatus  # success, failed, abandoned
    executed_steps: list[OperationOutcome]
    resolution: Resolution
    resolve_at: datetime

    def summarize(self, *, max_result_len: int = 300, max_response_len: int = 500) -> str:
        """Produce a multi-line, prompt-ready summary of this outcome.

        This keeps knowledge of the internal structure (Goal, OperationOutcome,
        StepStatus, etc.) confined to the ``agenturn`` package so that
        consumers (e.g. CLI interaction history) don't need to traverse the
        nested objects themselves.
        """
        parts: list[str] = []
        parts.append(
            f"  → Goal: {self.goal.description} | "
            f"Status: {self.status.value} | "
            f"Steps: {len(self.executed_steps)}"
        )
        if self.status.value != "completed" and self.reason:
            parts.append(f"  → Reason: {self.reason}")
        if self.executed_steps:
            parts.append("  Steps:")
            for i, step in enumerate(self.executed_steps, 1):
                tool_name = step.operation.tool or "N/A"
                status_symbol = "✓" if step.status.name == "SUCCESS" else "✗"
                reason = step.operation.reason
                parts.append(f"    {i}. {status_symbol} [{tool_name}] {reason}")
                result = step.result or step.error_message or "No result"
                if len(result) > max_result_len:
                    result = result[:max_result_len] + "…"
                parts.append(f"       Result: {result}")
        if self.response:
            resp = self.response
            if len(resp) > max_response_len:
                resp = resp[:max_response_len] + "…"
            parts.append(f"  → Response: {resp}")
        return "\n".join(parts)


@dataclass(frozen=True)
class PursuitProgress:
    """Tracks the progress of a goal pursuit."""
    goal: Goal
    executed_steps: list[OperationOutcome] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the pursuit progress."""
        lines = [f"Goal: {self.goal}"]

        # Show all executed steps (both completed and failed)
        if self.executed_steps:
            lines.append("Executed Steps:")
            for i, outcome in enumerate(self.executed_steps, 1):
                tool_name = outcome.operation.tool or "N/A"
                status_symbol = "✓" if outcome.status.name == "SUCCESS" else "✗"
                result = outcome.result or outcome.error_message or "No result"
                lines.append(f"  {i}. {status_symbol} [{tool_name}] {result}")

        return "\n".join(lines)


class ActionDeterminer(Protocol):
    async def determine_action(
            self,
            beliefs: list[str],
            pursuit_progress: PursuitProgress,
            available_tools: dict[str, SchematicTool],
            ctx: ExecutionContext,
            interaction_history: InteractionContext | None = None,
    ) -> OperationPlan | Resolution:
        ...


class PursuitAssessment(BaseModel):
    """Assessment of current pursuit progress toward a goal."""
    finished_tasks: Annotated[list[str], Field(description="List of tasks that have been completed toward the goal")]
    remaining_work_summary: Annotated[str, Field(description="Summary of what still needs to be done to achieve the goal")]
    required_context: Annotated[str, Field(description="Context still needed to complete the goal")]
    expected_actions: Annotated[str, Field(description="Actions expected to complete the goal")]
    boundary_conditions: Annotated[list[str], Field(description="Constraints for the remaining work")]
    exception_conditions: Annotated[list[str], Field(description="Edge cases or error conditions to handle going forward")]
    success_criteria: Annotated[list[str], Field(description="Conditions that indicate the goal is achieved")]


class PursuitAssessor(Protocol):
    async def assess_progress(
            self,
            pursuit: PursuitProgress,
            beliefs: list[str] | None = None,
            previous_assessment: PursuitAssessment | None = None,
            interaction_history: InteractionContext | None = None,
    ) -> PursuitAssessment:
        ...


class LLMPursuitAssessor:

    TEMPLATE_NAME = "assess_pursuit_progress.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en", lang_directive: str = ""):
        self.chat_llm = chat_llm.with_structured_output(PursuitAssessment)
        self._lang_directive = lang_directive
        template_env = TemplateEnvironment(package_name="novelrag.agenturn", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME)

    @trace_llm("pursuit_assessment")
    async def assess_progress(
            self,
            pursuit: PursuitProgress,
            beliefs: list[str] | None = None,
            previous_assessment: PursuitAssessment | None = None,
            interaction_history: InteractionContext | None = None,
    ) -> PursuitAssessment:
        """Assess the current progress of a pursuit toward its goal.
        
        Args:
            pursuit: The current pursuit progress with goal and executed steps
            beliefs: Optional list of agent beliefs (restrictions and guidelines)
                that should guide the assessment.
            previous_assessment: The assessment from the previous iteration, if any
            interaction_history: Optional recent interaction history for richer context
            
        Returns:
            A PursuitAssessment summarizing progress and remaining work
        """
        history_text = interaction_history.format_recent(3) if interaction_history else ""
        prompt = self.template.render(
            pursuit=pursuit,
            beliefs=beliefs or [],
            previous_assessment=previous_assessment,
            interaction_history=history_text,
        )

        response = await self.chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Assess the current pursuit progress.")
        ])
        assert isinstance(response, PursuitAssessment), "Expected PursuitAssessment from LLM response"
        return response
