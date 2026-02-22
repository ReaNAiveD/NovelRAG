"""Type definitions, enums, and data models for the agent package."""

from enum import Enum
from typing import Protocol, runtime_checkable


class AgentMessageLevel(str, Enum):
    """Message levels for agent outputs"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@runtime_checkable
class InteractionContext(Protocol):
    """Protocol for interaction history used by agent components.

    Provides a formatted summary of recent session interactions so that
    goal translators, goal deciders, and pursuit assessors can incorporate
    conversational context into their prompts without depending on the
    concrete CLI-layer ``InteractionHistory`` class.
    """

    def format_recent(self, n: int = 5) -> str:
        """Render the most recent *n* interactions as a prompt-ready string."""
        ...
