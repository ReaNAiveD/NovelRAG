"""Agent package for orchestrating tool execution and managing state.

This package provides a comprehensive agent system for managing tool execution,
scheduling, and coordination in the NovelRAG framework.
"""

# Main agent class
from .agent import Agent

# Core tool classes and mixins
from .tool import BaseTool, LLMToolMixin, SchematicTool, ContextualTool

# Tool adapters
from .adapters import SchematicToolAdapter

# Scheduling system
from .schedule import Step, Scheduler

# Proposal system
from .proposals import (
    ProposalSelector,
    TargetProposal,
    TargetProposer,
    ContentProposal,
    ContentProposer,
)

# Resource tools
from .resource_tools import (
    ResourceQueryTool,
    ResourceWriteTool,
    ResourceTypeCreateTool,
)

# Type definitions and enums
from .types import (
    # Message and output levels
    MessageLevel,
    AgentMessageLevel,
    
    # Tool output types
    ToolOutputType,
    ToolOutput,
    ToolMessage,
    ToolConfirmation,
    ToolResult,
    ToolUserInput,
    ToolStepProgress,
    ToolStepDecomposition,
    ToolBacklogOutput,
    
    # Agent output types
    AgentOutputType,
    AgentOutput,
    AgentMessage,
    AgentConfirmation,
    AgentUserInput,
    AgentResult,
    
    # Step status
    StepStatus,
    
    # Validation functions
    validate_tool_output,
    validate_tool_output_json,
)

__all__ = [
    # Main agent
    "Agent",
    
    # Tool system
    "BaseTool",
    "LLMToolMixin", 
    "SchematicTool",
    "ContextualTool",
    "SchematicToolAdapter",
    
    # Scheduling
    "Step",
    "Scheduler",
    
    # Proposals
    "ProposalSelector",
    "TargetProposal",
    "TargetProposer", 
    "ContentProposal",
    "ContentProposer",
    
    # Resource tools
    "ResourceQueryTool",
    "ResourceWriteTool",
    "ResourceTypeCreateTool",
    
    # Types and enums
    "MessageLevel",
    "AgentMessageLevel",
    "ToolOutputType",
    "ToolOutput",
    "ToolMessage",
    "ToolConfirmation",
    "ToolResult",
    "ToolUserInput",
    "ToolStepProgress",
    "ToolStepDecomposition",
    "ToolBacklogOutput",
    "AgentOutputType",
    "AgentOutput",
    "AgentMessage",
    "AgentConfirmation",
    "AgentUserInput",
    "AgentResult",
    "StepStatus",
    "validate_tool_output",
    "validate_tool_output_json",
]