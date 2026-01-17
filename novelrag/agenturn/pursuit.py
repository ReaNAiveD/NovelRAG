"""Data structures for goal pursuit and execution tracking."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.tool.schematic import SchematicTool
from novelrag.llm.mixin import LLMMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .step import OperationOutcome, OperationPlan, Resolution


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


@dataclass(frozen=True)
class PursuitProgress:
    """Tracks the progress of a goal pursuit."""
    goal: Goal
    pending_steps: list[str] = field(default_factory=list)
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
                lines.append(f"  {i}. {status_symbol} [{tool_name}] {outcome.operation.reason}")

        # Show all pending steps
        if self.pending_steps:
            lines.append("Pending Steps:")
            start_num = len(self.executed_steps) + 1
            for i, step in enumerate(self.pending_steps, start_num):
                lines.append(f"  {i}. ○ {step}")

        return "\n".join(lines)


class ActionDeterminer(Protocol):
    async def determine_action(
            self,
            beliefs: list[str],
            pursuit_progress: PursuitProgress,
            available_tools: dict[str, SchematicTool]
    ) -> OperationPlan | Resolution:
        ...


@dataclass
class PursuitAssessment:
    """Assessment of current pursuit progress toward a goal."""
    finished_tasks: list[str]
    remaining_work_summary: str
    required_context: str
    expected_actions: str
    boundary_conditions: list[str]
    exception_conditions: list[str]
    success_criteria: list[str]


PURSUIT_ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "finished_tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tasks that have been completed toward the goal"
        },
        "remaining_work_summary": {
            "type": "string",
            "description": "Summary of what still needs to be done to achieve the goal"
        },
        "required_context": {
            "type": "string",
            "description": "Context still needed to complete the goal"
        },
        "expected_actions": {
            "type": "string",
            "description": "Actions expected to complete the goal"
        },
        "boundary_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Constraints for the remaining work"
        },
        "exception_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Edge cases or error conditions to handle going forward"
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions that indicate the goal is achieved"
        }
    },
    "required": [
        "finished_tasks",
        "remaining_work_summary",
        "required_context",
        "expected_actions",
        "boundary_conditions",
        "exception_conditions",
        "success_criteria"
    ],
    "additionalProperties": False
}


class PursuitAssessor(LLMMixin):

    TEMPLATE_NAME = "assess_pursuit_progress.jinja2"

    def __init__(self, chat_llm: ChatLLM, lang: str = "en"):
        template_env = TemplateEnvironment(package_name="novelrag.agenturn", default_lang=lang)
        super().__init__(template_env, chat_llm)

    async def assess_progress(
            self,
            pursuit: PursuitProgress,
            beliefs: list[str] | None = None,
            previous_assessment: PursuitAssessment | None = None
    ) -> PursuitAssessment:
        """Assess the current progress of a pursuit toward its goal.
        
        Args:
            pursuit: The current pursuit progress with goal and executed steps
            beliefs: Optional list of agent beliefs (restrictions and guidelines)
                that should guide the assessment.
            previous_assessment: The assessment from the previous iteration, if any
            
        Returns:
            A PursuitAssessment summarizing progress and remaining work
        """
        response = await self.call_template_structured(
            template_name=self.TEMPLATE_NAME,
            response_schema=PURSUIT_ASSESSMENT_SCHEMA,
            user_question="Assess the current pursuit progress.",
            pursuit=pursuit,
            beliefs=beliefs or [],
            previous_assessment=previous_assessment
        )

        data = json.loads(response)
        
        return PursuitAssessment(
            finished_tasks=data["finished_tasks"],
            remaining_work_summary=data["remaining_work_summary"],
            required_context=data["required_context"],
            expected_actions=data["expected_actions"],
            boundary_conditions=data["boundary_conditions"],
            exception_conditions=data["exception_conditions"],
            success_criteria=data["success_criteria"]
        )
