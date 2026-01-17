"""Schematic tool with input schema support.

This module provides the SchematicTool class which extends BaseTool
with structured input schema capabilities.
"""

from abc import abstractmethod
from typing import Any

from novelrag.agent.workspace import ResourceContext

from .base import BaseTool
from .runtime import ToolRuntime
from .types import ToolOutput


class SchematicTool(BaseTool):
    """Tool with structured input schema support"""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Schema for input parameters required by the tool"""
        pass

    @abstractmethod
    async def call(self, runtime: ToolRuntime, context: ResourceContext, **kwargs) -> ToolOutput:
        """Execute the tool with provided parameters and yield outputs asynchronously"""
        raise NotImplementedError("SchematicTool subclasses must implement the call method")
