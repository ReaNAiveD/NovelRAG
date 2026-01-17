"""Core step definitions and data structures for agent execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StepStatus(Enum):
    """Status of an action's execution."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OperationPlan:
    """Represents the core definition of a step - immutable tool and intent description."""
    reason: str
    tool: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Resolution:
    reason: str
    response: str
    status: str  # success, failed, abandoned


@dataclass
class OperationOutcome:
    """The result of executing an action."""
    operation: OperationPlan
    status: StepStatus
    results: list[str] = field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
