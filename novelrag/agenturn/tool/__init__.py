"""Tool subpackage for agent tool interfaces.

This package provides the core tool abstractions:
- BaseTool: Abstract base class for all tools
- SchematicTool: Tool with structured input schema support

Resource-specific tools have been moved to `novelrag.resource_agent.tool`.
"""

from .base import BaseTool
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
    
    # Tool output types
    "ToolOutput",
    "ToolOutputType",
    "ToolError",
    "ToolResult",
    "validate_tool_output",
    "validate_tool_output_json",
]
