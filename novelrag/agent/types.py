"""Type definitions, enums, and data models for the agent package."""

from typing import Any, Literal, Annotated
from enum import Enum
from pydantic import BaseModel, Field, TypeAdapter, ConfigDict


class MessageLevel(str, Enum):
    """Message levels for user-visible logs"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ToolOutputType(str, Enum):
    """Types of tool outputs"""
    MESSAGE = "message"
    CONFIRMATION = "confirmation"
    OUTPUT = "output"
    USER_INPUT = "user_input"
    STEP_PROGRESS = "step_progress"
    STEP_DECOMPOSITION = "step_decomposition"
    OUTPUT_WITH_REPLANNING = "output_with_replanning"
    BACKLOG = "backlog"


class ToolOutputBase(BaseModel):
    """Base class for all tool outputs"""
    model_config = ConfigDict(extra="forbid")
    
    type: Annotated[ToolOutputType, Field(description="Type of the output")]


class ToolMessage(ToolOutputBase):
    """Message to display to the user"""
    type: Literal[ToolOutputType.MESSAGE] = ToolOutputType.MESSAGE  # type: ignore
    content: Annotated[str, Field(description="Message content")]
    level: Annotated[MessageLevel, Field(default=MessageLevel.INFO, description="Message level")]


class ToolConfirmation(ToolOutputBase):
    """Sensitive operation requiring user confirmation"""
    type: Literal[ToolOutputType.CONFIRMATION] = ToolOutputType.CONFIRMATION # type: ignore
    prompt: Annotated[str, Field(description="Confirmation prompt text")]


class ToolResult(ToolOutputBase):
    """Output result of current step, managed by framework"""
    type: Literal[ToolOutputType.OUTPUT] = ToolOutputType.OUTPUT # type: ignore
    result: Annotated[str, Field(description="Result data")]


class ToolUserInput(ToolOutputBase):
    """Request for specific user input"""
    type: Literal[ToolOutputType.USER_INPUT] = ToolOutputType.USER_INPUT # type: ignore
    prompt: Annotated[str, Field(description="Input prompt text")]


class ToolStepProgress(ToolOutputBase):
    """Update progress or intermediate state of the current step"""
    type: Literal[ToolOutputType.STEP_PROGRESS] = ToolOutputType.STEP_PROGRESS # type: ignore
    field: Annotated[str, Field(description="Field name to update in the step")]
    value: Annotated[Any, Field(description="Value to set for the field")]
    description: Annotated[str | None, Field(default=None, description="Description of this progress update")]


class ToolStepDecomposition(ToolOutputBase):
    """Tool directly decomposes into further planning steps"""
    type: Literal[ToolOutputType.STEP_DECOMPOSITION] = ToolOutputType.STEP_DECOMPOSITION # type: ignore
    steps: Annotated[list[dict[str, str]], Field(description="List of decomposed steps")]
    rationale: Annotated[str | None, Field(default=None, description="Reasoning for decomposition")]


class ToolBacklogOutput(ToolOutputBase):
    """Add specified content to backlog"""
    type: Literal[ToolOutputType.BACKLOG] = ToolOutputType.BACKLOG # type: ignore
    content: Annotated[str, Field(description="Content to add to backlog")]
    priority: Annotated[str | None, Field(default=None, description="Priority level in backlog")]


# Union type for all possible outputs
ToolOutput = Annotated[(
    ToolMessage |
    ToolConfirmation |
    ToolResult |
    ToolUserInput |
    ToolStepProgress |
    ToolStepDecomposition |
    ToolBacklogOutput
), Field(discriminator='type')]


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


class AgentOutputType(str, Enum):
    """Types of agent outputs"""
    MESSAGE = "message"
    CONFIRMATION = "confirmation"
    USER_INPUT = "user_input"
    RESULT = "result"


class AgentOutputBase(BaseModel):
    """Base class for all agent outputs"""
    model_config = ConfigDict(extra="forbid")


class AgentMessage(AgentOutputBase):
    """Message to display to the user"""
    type: Literal[AgentOutputType.MESSAGE] = AgentOutputType.MESSAGE
    content: Annotated[str, Field(description="Message content")]
    level: Annotated[AgentMessageLevel, Field(default=AgentMessageLevel.INFO, description="Message level")]


class AgentConfirmation(AgentOutputBase):
    """Sensitive operation requiring user confirmation"""
    type: Literal[AgentOutputType.CONFIRMATION] = AgentOutputType.CONFIRMATION
    prompt: Annotated[str, Field(description="Confirmation prompt text")]


class AgentUserInput(AgentOutputBase):
    """Request for specific user input"""
    type: Literal[AgentOutputType.USER_INPUT] = AgentOutputType.USER_INPUT
    prompt: Annotated[str, Field(description="Input prompt text")]


class AgentResult(AgentOutputBase):
    """Output result of current step, managed by framework"""
    type: Literal[AgentOutputType.RESULT] = AgentOutputType.RESULT
    result: Annotated[Any, Field(description="Output data")]


# Union type for all possible agent outputs
AgentOutput = Annotated[
    AgentMessage | AgentConfirmation | AgentUserInput | AgentResult,
    Field(discriminator='type')
]


def validate_agent_output(data: dict) -> AgentOutput:
    """Validate and return an AgentOutput instance from raw data."""
    return TypeAdapter(AgentOutput).validate_python(data)


def validate_agent_output_json(data: str) -> AgentOutput:
    """Validate and return an AgentOutput instance from JSON string."""
    return TypeAdapter(AgentOutput).validate_json(data)
