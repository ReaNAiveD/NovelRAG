"""Agent package for orchestrating tool execution and managing state.

This package provides a comprehensive agent system for managing tool execution,
scheduling, and coordination in the NovelRAG framework.
"""

# Main agent class
from .agent import Agent

# Communication channels
from .channel import AgentChannel, SessionChannel, ShellSessionChannel

# Core tool interfaces for extension
from .tool import BaseTool, SchematicTool, ContextualTool

# Essential types for public API
from .types import (
    # Core enums
    MessageLevel,
    AgentMessageLevel,
    ToolOutputType,
    AgentOutputType,

    # Main union types
    ToolOutput,
    AgentOutput,

    # Validation functions
    validate_tool_output,
    validate_tool_output_json,
    validate_agent_output,
)

__all__ = [
    # Main agent
    "Agent",
    
    # Communication channels
    "AgentChannel",
    "SessionChannel",
    "ShellSessionChannel",

    # Core tool interfaces
    "BaseTool",
    "SchematicTool",
    "ContextualTool",

    # Essential types
    "MessageLevel",
    "AgentMessageLevel",
    "ToolOutputType",
    "AgentOutputType",
    "ToolOutput",
    "AgentOutput",

    # Validation functions
    "validate_tool_output",
    "validate_tool_output_json",
    "validate_agent_output",
]