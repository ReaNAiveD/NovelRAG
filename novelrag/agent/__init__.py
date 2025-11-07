"""Agent package for orchestrating tool execution and managing state.

This package provides a comprehensive agent system for managing tool execution,
scheduling, and coordination in the NovelRAG framework.
"""

# Main agent class and factory
from .agent import Agent, create_agent

# Communication channels
from .channel import AgentChannel, SessionChannel, ShellSessionChannel

# Core tool interfaces for extension
from .tool import BaseTool, SchematicTool, ToolRuntime

# Pursuit data structures
from .pursuit_types import GoalPursuitResult, PursuitOutcome, PursuitProgress, PursuitStatus

# Essential types for public API
from .types import (
    ToolOutputType,

    # Main union types
    ToolOutput,

    # Validation functions
    validate_tool_output,
    validate_tool_output_json,

    AgentMessageLevel,
)

__all__ = [
    # Main agent
    "Agent",
    "create_agent",
    
    # Communication channels
    "AgentChannel",
    "SessionChannel",
    "ShellSessionChannel",

    # Core tool interfaces
    "BaseTool",
    "SchematicTool",
    "ToolRuntime",

    # Pursuit data structures
    "GoalPursuitResult",
    "PursuitOutcome", 
    "PursuitProgress",
    "PursuitStatus",

    # Essential types
    "ToolOutputType",
    "ToolOutput",
    "AgentMessageLevel",

    # Validation functions
    "validate_tool_output",
    "validate_tool_output_json",
]