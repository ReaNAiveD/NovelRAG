"""Tool adapters for converting between different tool interfaces."""

import json
from typing import AsyncGenerator, Any

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .tool import LLMToolMixin, ContextualTool, SchematicTool
from .types import ToolOutput


class SchematicToolAdapter(LLMToolMixin, ContextualTool):
    """Adapts a SchematicTool to work as a ContextualTool with LLM argument building."""
    
    def __init__(self, schematic_tool: SchematicTool, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.inner = schematic_tool
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    @property
    def name(self) -> str:
        """Name of the tool, delegated to the inner tool"""
        return self.inner.name

    @property
    def description(self) -> str | None:
        """Description of the tool, delegated to the inner tool"""
        return self.inner.description

    @property
    def output_description(self) -> str | None:
        """Output description of the tool, delegated to the inner tool"""
        return self.inner.output_description

    async def call(self, believes: list[str] | None = None, step_description: str | None = None, context: list[str] | None = None, tools: dict[str, str] | None = None) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Execute the tool with contextual information and yield outputs asynchronously"""
        # Check if we need to gather additional context for missing inputs
        missing_fields = await self._identify_missing_inputs(believes, step_description, context)
        
        if missing_fields and self.inner.requires_prerequisite_steps:
            # Tool can decompose into steps to gather missing data
            steps = await self._build_data_gathering_steps(missing_fields, tools)
            # If we failed to build steps, we have to work with what we have
            if steps:
                yield self.step_decomposition(
                    steps=steps,
                    rationale=f"Need to gather missing inputs: {', '.join(missing_fields)}"
                )
                return
            yield self.warning(f"Failed to build steps for missing inputs: {', '.join(missing_fields)}. Proceeding with available context.")
        elif missing_fields:
            # Tool cannot decompose, must work with what we have
            yield self.warning(f"Missing inputs: {', '.join(missing_fields)}. Proceeding with available context.")
        
        # Build arguments from context using LLM
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tool_args = await self._build_tool_arguments(believes, step_description, context)
            except Exception as e:
                yield self.warning(f"Error building tool arguments: {str(e)}")
                tool_args = None
            if tool_args is not None:
                break
            if attempt < max_retries - 1:
                yield self.warning(f"Failed to build tool arguments (attempt {attempt + 1}/{max_retries}). Retrying...")
        else:
            yield self.error(f"Failed to build tool arguments after {max_retries} attempts")
            return
        try:
            # Call the inner schematic tool with the built arguments
            async for output in self.inner.call(**tool_args):
                yield output
        except Exception as e:
            yield self.error(f"Failed to execute schematic tool {self.inner.name}: {str(e)}")

    async def _identify_missing_inputs(self, believes: list[str] | None = None, step_description: str | None = None, context: list[str] | None = None) -> list[str]:
        """Identify which required inputs are missing from the current context"""
        input_schema = self.inner.input_schema
        
        # Use LLM to check the missing inputs
        # Also identify vital optional properties in input schema
        missing_info = await self.call_template(
            "identify_missing_inputs.jinja2",
            tool_name=self.inner.name,
            tool_description=self.inner.description or "No description available",
            step_description=step_description or "",
            context=context or [],
            believes=believes or [],
            input_schema=input_schema
        )
        
        missing_fields = missing_info.splitlines()
        return missing_fields

    async def _build_data_gathering_steps(self, missing_fields: list[str], tools: dict[str, str] | None = None) -> list[dict[str, str]]:
        """Build steps to gather missing data using available tools"""
        if not tools:
            return []
        
        steps_plan = await self.call_template(
            "build_data_gathering_steps.jinja2",
            missing_fields=missing_fields,
            available_tools=tools,
            input_schema=self.inner.input_schema,
            json_format=True
        )
        
        # Parse the JSON response to get steps
        try:
            return json.loads(steps_plan)['steps']
        except json.JSONDecodeError:
            return []

    async def _build_tool_arguments(self, believes: list[str] | None = None, step_description: str | None = None, context: list[str] | None = None) -> dict[str, Any] | None:
        """Build tool arguments from context using LLM"""
        input_schema = self.inner.input_schema
        
        arguments_json = await self.call_template(
            "build_tool_arguments.jinja2",
            tool_name=self.inner.name,
            tool_description=self.inner.description or "No description available",
            step_description=step_description or "",
            context=context or [],
            believes=believes or [],
            input_schema=input_schema,
            json_format=True
        )
        
        # Parse the JSON response to get arguments
        try:
            return json.loads(arguments_json)
        except json.JSONDecodeError:
            return None
