"""Abstract base class for all tools.

This module provides the foundational base class that all tools must inherit from.
"""

from abc import ABC, abstractmethod

from .types import ToolResult, ToolError


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
