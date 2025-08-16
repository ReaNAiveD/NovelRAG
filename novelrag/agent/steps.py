"""Core step definitions and data structures for agent execution."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StepStatus(Enum):
    """Status of an action's execution."""
    SUCCESS = "success"
    FAILED = "failed"
    DECOMPOSED = "decomposed"  # Action was broken down into sub-actions
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class StepDefinition:
    """Represents the core definition of a step - immutable tool and intent description."""
    tool: str | None
    intent: str  # What the agent intends to achieve with this action
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    progress: dict[str, list[str]] = field(default_factory=dict)  # Progress for recovery and tracking


@dataclass
class ExecutableStep:
    """Represents an executable step with relationships and execution capability."""
    definition: StepDefinition

    # Relationships
    contribute_to: 'ExecutableStep | None' = None  # Step who depends on the results of this step
    spawned_by: 'StepOutcome | None' = None  # Parent action that decomposed into this
    triggered_by: 'StepOutcome | None' = None  # Action that triggered this as follow-up

    @property
    def tool(self) -> str:
        """Access to the tool from the definition."""
        return self.definition.tool

    @property
    def intent(self) -> str:
        """Access to the intent from the definition."""
        return self.definition.intent

    @property
    def step_id(self) -> str:
        """Access to the step_id from the definition."""
        return self.definition.step_id


@dataclass
class StepOutcome:
    """The result of executing an action."""
    action: ExecutableStep
    status: StepStatus
    results: list[str] = field(default_factory=list)
    progress: dict[str, list[str]] = field(default_factory=dict)
    error_message: str | None = None

    # Dynamic plan modifications
    spawned_actions: list[ExecutableStep] = field(default_factory=list)  # From decomposition
    triggered_actions: list[ExecutableStep] = field(default_factory=list)  # From chain updates

    # Discovered insights and future work
    discovered_insights: list[str] = field(default_factory=list)
    backlog_items: list[str] = field(default_factory=list)

    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None
