"""Base classes for tools and mixins."""

from abc import ABC, abstractmethod
import json
from typing import Any, AsyncGenerator

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .types import MessageLevel, ToolOutput, ToolMessage, ToolConfirmation, ToolResult, ToolUserInput, ToolStepProgress, ToolStepDecomposition, ToolBacklogOutput


class LLMToolMixin:
    """Mixin that provides LLM template calling functionality"""
    
    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.template_env = template_env
        self.chat_llm = chat_llm
    
    async def call_template(self, template_name: str, user_question: str | None = None, json_format: bool = False, **kwargs: str | list | dict) -> str:
        """Call an LLM with a template and return the response."""
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        return await self.chat_llm.chat(messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please answer the question.'}
        ], response_format='json_object' if json_format else None)


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

    def debug(self, content: str) -> ToolMessage:
        """Create a debug Message output"""
        return ToolMessage(content=content, level=MessageLevel.DEBUG)

    def message(self, content: str, level: MessageLevel = MessageLevel.INFO) -> ToolMessage:
        """Create a Message output"""
        return ToolMessage(content=content, level=level)
    
    def warning(self, content: str) -> ToolMessage:
        """Create a warning Message output"""
        return ToolMessage(content=content, level=MessageLevel.WARNING)
    
    def error(self, content: str) -> ToolMessage:
        """Create an error Message output"""
        return ToolMessage(content=content, level=MessageLevel.ERROR)

    def confirmation(self, prompt: str) -> ToolConfirmation:
        """Create a Confirmation output"""
        return ToolConfirmation(prompt=prompt)

    def output(self, output: str) -> ToolResult:
        """Create an Output result"""
        return ToolResult(result=output)

    def user_input(self, prompt: str) -> ToolUserInput:
        """Create a UserInput request"""
        return ToolUserInput(prompt=prompt)
    
    def step_progress(self, field: str, value: Any, description: str | None = None) -> ToolStepProgress:
        """Create a StepProgress output"""
        return ToolStepProgress(field=field, value=value, description=description)

    def step_decomposition(self, steps: list[dict[str, str]], rationale: str | None = None) -> ToolStepDecomposition:
        """Create a StepDecomposition output"""
        return ToolStepDecomposition(steps=steps, rationale=rationale)

    def backlog(self, content: Any, priority: str | None = None) -> ToolBacklogOutput:
        """Create a BacklogOutput"""
        return ToolBacklogOutput(content=content, priority=priority)


class SchematicTool(BaseTool):
    @property
    def requires_prerequisite_steps(self) -> bool:
        """Whether this tool may require prerequisite steps to gather additional data
        when the provided context is insufficient for its input schema."""
        return False

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Schema for input parameters required by the tool"""
        pass

    @abstractmethod
    async def call(self, **kwargs) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Execute the tool with provided parameters and yield outputs asynchronously"""
        yield self.error("Tool call not implemented")

    def wrapped(self, template_env: TemplateEnvironment, chat_llm: ChatLLM) -> 'SchematicToolAdapter':
        return SchematicToolAdapter(self, template_env, chat_llm)


class ContextualTool(BaseTool):
    @abstractmethod
    async def call(self, believes: list[str] | None = None, step_description: str | None = None, context: list[str] | None = None, tools: dict[str, str] | None = None) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Execute the tool with contextual information and yield outputs asynchronously"""
        yield self.error("Tool call not implemented")


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
