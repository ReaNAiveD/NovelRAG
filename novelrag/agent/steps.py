"""Core step definitions and data structures for agent execution."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StepStatus(Enum):
    """Status of an action's execution."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class StepDefinition:
    """Represents the core definition of a step - immutable tool and intent description."""
    reason: str
    tool: str
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parameters: dict = field(default_factory=dict)


@dataclass
class StepOutcome:
    """The result of executing an action."""
    action: StepDefinition
    status: StepStatus
    results: list[str] = field(default_factory=list)
    progress: dict[str, list[str]] = field(default_factory=dict)
    error_message: str | None = None

    # Dynamic plan modifications
    triggered_actions: list[dict[str, str]] = field(default_factory=list)  # From chain updates

    # Discovered insights and future work
    discovered_insights: list[str] = field(default_factory=list)
    backlog_items: list[str] = field(default_factory=list)

    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None
