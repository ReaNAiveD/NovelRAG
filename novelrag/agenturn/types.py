"""Type definitions, enums, and data models for the agent package."""

from enum import Enum


class AgentMessageLevel(str, Enum):
    """Message levels for agent outputs"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
