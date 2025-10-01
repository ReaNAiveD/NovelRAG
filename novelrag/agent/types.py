"""Type definitions, enums, and data models for the agent package."""

from typing import Literal, Annotated
from enum import Enum
from pydantic import BaseModel, Field, TypeAdapter, ConfigDict


class ToolOutputType(str, Enum):
    """Types of tool outputs"""
    OUTPUT = "output"
    ERROR = "error"


class ToolOutputBase(BaseModel):
    """Base class for all tool outputs"""
    model_config = ConfigDict(extra="forbid")
    
    type: Annotated[ToolOutputType, Field(description="Type of the output")]


class ToolError(ToolOutputBase):
    """Error encountered during tool execution"""
    type: Literal[ToolOutputType.ERROR] = ToolOutputType.ERROR # type: ignore
    error_message: Annotated[str, Field(description="Error message")]


class ToolResult(ToolOutputBase):
    """Output result of current step, managed by framework"""
    type: Literal[ToolOutputType.OUTPUT] = ToolOutputType.OUTPUT # type: ignore
    result: Annotated[str, Field(description="Result data")]


# Union type for all possible outputs
ToolOutput = Annotated[
    ToolResult | ToolError,
    Field(discriminator='type')
]


def validate_tool_output(data: dict) -> ToolOutput:
    """Validate and return a ToolOutput instance from raw data."""
    return TypeAdapter(ToolOutput).validate_python(data)


def validate_tool_output_json(data: str) -> ToolOutput:
    """Validate and return a ToolOutput instance from JSON string."""
    return TypeAdapter(ToolOutput).validate_json(data)


class AgentMessageLevel(str, Enum):
    """Message levels for agent outputs"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
