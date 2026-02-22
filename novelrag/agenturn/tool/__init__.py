"""Tool subpackage for agent tool interfaces.

This package provides the core tool abstractions:
- BaseTool: Abstract base class for all tools
- SchematicTool: Tool with structured input schema support
- ToolRuntime: Deprecated alias for ExecutionContext (kept for backward compatibility)

Resource-specific tools have been moved to `novelrag.resource_agent.tool`.
"""

from .base import BaseTool
from .runtime import ToolRuntime
from .schematic import SchematicTool
from .types import (
    ToolOutput,
    ToolOutputType,
    ToolError,
    ToolResult,
    validate_tool_output,
    validate_tool_output_json,
)

__all__ = [
    # Core abstractions
    "BaseTool",
    "SchematicTool",
    "ToolRuntime",
    
    # Tool output types
    "ToolOutput",
    "ToolOutputType",
    "ToolError",
    "ToolResult",
    "validate_tool_output",
    "validate_tool_output_json",
]
