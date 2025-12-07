"""Data structures for goal pursuit and execution tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from novelrag.llm import LLMMixin

from .steps import StepOutcome


class GoalBuilder(LLMMixin):
    async def build_goal(self, user_request: str) -> str:
        """Build a clear and concise goal from the user's request."""
        goal = await self.call_template(
            "translate_request_to_goal.jinja2",
            user_request=user_request
        )
        if goal.startswith("**Goal**: "):
            goal = goal[len("**Goal**: "):]
        return goal.strip()


class PursuitStatus(Enum):
    """Status of a goal pursuit."""
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class GoalPursuitResult:
    """Represents the result of pursuing a specific goal."""
    goal: str
    status: PursuitStatus
    executed_steps: list[StepOutcome] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PursuitOutcome:
    """Represents the final outcome of a goal pursuit."""
    goal: str
    reason: str  # Why goal pursuit ended
    response: str  # User-facing completion message
    status: str  # success, failed, abandoned
    executed_steps: list[StepOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class PursuitProgress:
    """Tracks the progress of a goal pursuit."""
    goal: str
    pending_steps: list[str] = field(default_factory=list)
    executed_steps: list[StepOutcome] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the pursuit progress."""
        lines = [f"Goal: {self.goal}"]

        # Show all executed steps (both completed and failed)
        if self.executed_steps:
            lines.append("Executed Steps:")
            for i, outcome in enumerate(self.executed_steps, 1):
                tool_name = outcome.action.tool or "N/A"
                status_symbol = "✓" if outcome.status.name == "SUCCESS" else "✗"
                lines.append(f"  {i}. {status_symbol} [{tool_name}] {outcome.action.reason}")

        # Show all pending steps
        if self.pending_steps:
            lines.append("Pending Steps:")
            start_num = len(self.executed_steps) + 1
            for i, step in enumerate(self.pending_steps, start_num):
                lines.append(f"  {i}. ○ {step}")

        return "\n".join(lines)