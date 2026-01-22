"""Agent package for orchestrating tool execution and managing state.

This package provides a generic agent framework for goal pursuit, tool execution,
and coordination. Resource-specific agent logic is in the `resource_agent` package.

Core components:
- GoalExecutor: Executes a single goal using tools and action determination
- AgentToolRuntime: Routes tool runtime calls to AgentChannel
- ActionDeterminer: Protocol for action determination strategies
- PursuitAssessor: Assesses progress toward goals

Communication:
- AgentChannel: Protocol for agent-user communication
- SessionChannel, ShellSessionChannel: Concrete channel implementations

Tool abstractions:
- BaseTool, SchematicTool: Tool interfaces
- ToolRuntime: Runtime interface for tool side-effects
"""

# Main agent class
from .agent import GoalExecutor, AgentToolRuntime

# Communication channels
from .channel import AgentChannel

# Goal abstractions
from .goal import Goal

# Action determination and pursuit assessment
from .pursuit import (
    ActionDeterminer,
    PursuitAssessor,
    PursuitAssessment,
    PursuitOutcome,
    PursuitProgress,
    PursuitStatus,
)

# Step definitions
from .step import OperationPlan, OperationOutcome, Resolution, StepStatus

# Core tool interfaces for extension
from .tool import BaseTool, SchematicTool, ToolRuntime

# Essential types for public API
from .tool.types import (
    ToolOutputType,
    ToolOutput,
    ToolResult,
    ToolError,
    validate_tool_output,
    validate_tool_output_json,
)
from .types import AgentMessageLevel

__all__ = [
    # Main agent
    "GoalExecutor",
    "AgentToolRuntime",
    
    # Communication channels
    "AgentChannel",

    # Goal abstractions
    "Goal",

    # Action determination and pursuit
    "ActionDeterminer",
    "PursuitAssessor",
    "PursuitAssessment",
    "PursuitOutcome", 
    "PursuitProgress",
    "PursuitStatus",

    # Step definitions
    "OperationPlan",
    "OperationOutcome",
    "Resolution",
    "StepStatus",

    # Core tool interfaces
    "BaseTool",
    "SchematicTool",
    "ToolRuntime",

    # Tool output types
    "ToolOutputType",
    "ToolOutput",
    "ToolResult",
    "ToolError",
    "AgentMessageLevel",

    # Validation functions
    "validate_tool_output",
    "validate_tool_output_json",
]