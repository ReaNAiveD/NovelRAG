"""Tool subpackage for agent tool interfaces.

This package provides the core tool abstractions:
- BaseTool: Abstract base class for all tools
- SchematicTool: Tool with structured input schema support
- ToolRuntime: Runtime interaction interface for tools

Query tools (read-only):
- ResourceFetchTool: Fetch resources by URI
- ResourceSearchTool: Semantic vector search

Write tools (modify resources):
- AspectCreateTool: Create new aspects
- ResourceWriteTool: Edit existing resources
- ResourceRelationWriteTool: Manage resource relations
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

# Query tools
from .query import ResourceFetchTool, ResourceSearchTool

# Write tools
from .write import (
    ContentGenerationTask,
    OperationPlan,
    AspectCreateTool,
    ResourceWriteTool,
    ResourceRelationWriteTool,
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
    
    # Query tools
    "ResourceFetchTool",
    "ResourceSearchTool",
    
    # Write tools
    "AspectCreateTool",
    "ResourceWriteTool",
    "ResourceRelationWriteTool",
    
    # Write tool types
    "ContentGenerationTask",
    "OperationPlan",
]
]
