"""Tool interfaces and output types for the agent system.

This module provides:
- BaseTool: Abstract base class for all tools
- SchematicTool: Tool with structured input schema support
- ToolOutput types: ToolResult, ToolError, and validation helpers
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ConfigDict
from typing_extensions import Literal, Annotated

from novelrag.agenturn.procedure import ExecutionContext


# ---------------------------------------------------------------------------
# Tool output types
# ---------------------------------------------------------------------------


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
    type: Literal[ToolOutputType.ERROR] = ToolOutputType.ERROR  # type: ignore
    error_message: Annotated[str, Field(description="Error message")]


class ToolResult(ToolOutputBase):
    """Output result of current step, managed by framework"""
    type: Literal[ToolOutputType.OUTPUT] = ToolOutputType.OUTPUT  # type: ignore
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


# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """Abstract base class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool, used for identification"""
        pass

    @property
    def description(self) -> str | None:
        """Description of what the tool does"""
        return None

    @property
    def output_description(self) -> str | None:
        """Description of the output format"""
        return None

    @property
    def prerequisites(self) -> str | None:
        """Any prerequisites or setup required before using the tool"""
        return None

    @staticmethod
    def result(result: str) -> ToolResult:
        return ToolResult(result=result)

    @staticmethod
    def error(error_message: str) -> ToolError:
        """Create a ToolError output with the given error message"""
        return ToolError(error_message=error_message)


# ---------------------------------------------------------------------------
# Schematic tool
# ---------------------------------------------------------------------------


class SchematicTool(BaseTool):
    """Tool with structured input schema support"""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Schema for input parameters required by the tool"""
        pass

    @abstractmethod
    async def call(self, ctx: ExecutionContext, **kwargs) -> ToolOutput:
        """Execute the tool with provided parameters and yield outputs asynchronously"""
        raise NotImplementedError("SchematicTool subclasses must implement the call method")


__all__ = [
    "BaseTool",
    "SchematicTool",
    "ToolOutput",
    "ToolOutputType",
    "ToolOutputBase",
    "ToolError",
    "ToolResult",
    "validate_tool_output",
    "validate_tool_output_json",
]
