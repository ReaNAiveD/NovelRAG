"""Base classes for tools and mixins."""

from abc import ABC, abstractmethod
import json
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment
from .steps import StepDefinition

from .types import ToolOutput, ToolDecomposition, ToolResult, ToolError


class LLMToolMixin:
    """Mixin that provides LLM template calling functionality"""
    
    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.template_env = template_env
        self.chat_llm = chat_llm
    
    async def call_template(self, template_name: str, user_question: str | None = None, json_format: bool = False, **kwargs: bool | int | float | str | list | dict) -> str:
        """Call an LLM with a template and return the response."""
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        return await self.chat_llm.chat(messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please answer the question.'}
        ], response_format='json_object' if json_format else None)


class ToolRuntime(ABC):
    """Runtime interaction interface for tools.

    Why this exists: as part of an upcoming refactor, tool.call will accept a
    ToolRuntime instance instead of returning an AsyncGenerator of ToolOutput.
    The tool will return its final output (or raise on error), while side-channel
    interactions such as debug/info/warnings, confirmations, user input, progress
    updates, and backlog management will be routed through this interface.

    This decouples tool I/O from return values, simplifies testing, and allows
    different runtime implementations (CLI, UI, logs, etc.).
    """

    @abstractmethod
    async def debug(self, content: str):
        """Emit a developer-focused debug message.

        Intended for verbose diagnostics not shown to end users. Implementations
        should avoid blocking and may buffer or drop if necessary. Async: must be
        awaited by callers.
        """
        pass

    @abstractmethod
    async def message(self, content: str):
        """Emit a user-visible message.

        Args:
        content: the message text to display

        Implementations should surface this in UI/logs. Non-blocking; async and
        should be awaited.
        """
        pass

    @abstractmethod
    async def warning(self, content: str):
        """Emit a user-visible warning about a recoverable condition.

        Use when execution can continue but attention is warranted. This does not
        raise; callers decide whether to proceed. Async and should be awaited.
        """
        pass

    @abstractmethod
    async def error(self, content: str):
        """Emit a user-visible error message describing a failure.

        This reports the error via the runtime side-channel but does not itself
        raise an exception. Tools may still raise to abort execution. Async and
        should be awaited.
        """
        pass

    @abstractmethod
    async def confirmation(self, prompt: str) -> bool:
        """Request an explicit yes/no confirmation from the user.

        Implementations should present the prompt and resolve to a truthy value to
        proceed and a falsy value to abort/skip. Async and must be awaited.
        Return semantics are implementation-defined but should behave like a bool.
        """
        pass

    @abstractmethod
    async def user_input(self, prompt: str) -> str:
        """Prompt the user for free-form input and return it.

        Implementations should collect input via UI/CLI and return the entered
        string (or equivalent). Async and must be awaited.
        """
        pass

    @abstractmethod
    async def progress(self, key: str, value: str, description: str | None = None):
        """Record a progress update for the current step.

        Parameters:
        - key: canonical progress key (e.g., 'downloaded_bytes')
        - value: value for this key appended to the list
        - description: optional human-readable note

        Implementations should persist/emit the update and be non-blocking when
        possible. Async and should be awaited.
        """
        pass

    @abstractmethod
    async def trigger_action(self, action: dict[str, str]):
        """Trigger a new action to be executed after the current one.

        Parameters:
        - action: action definition dict

        Implementations should enqueue the action for later execution in the same pursuit.
        Async and should be awaited.
        """
        pass

    @abstractmethod
    async def backlog(self, content: dict, priority: str | None = None):
        """Add an item to the backlog for later processing.

        Parameters:
        - content: arbitrary data or description of the follow-up work
        - priority: optional label (e.g., 'low' | 'normal' | 'high')

        Implementations should enqueue the item durably. Async and should be
        awaited.
        """
        pass


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

    @staticmethod
    def result(result: str) -> ToolResult:
        return ToolResult(result=result)

    @staticmethod
    def error(error_message: str) -> ToolError:
        """Create a ToolError output with the given error message"""
        return ToolError(error_message=error_message)

    @staticmethod
    def decomposition(steps: list[dict[str, str]], rationale: str | None = None, rerun: bool = False) -> ToolDecomposition:
        """Create a ToolDecomposition output with the given steps and rationale"""
        return ToolDecomposition(steps=steps, rationale=rationale, rerun=rerun)


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
    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Execute the tool with provided parameters and yield outputs asynchronously"""
        raise NotImplementedError("SchematicTool subclasses must implement the call method")

    def wrapped(self, template_env: TemplateEnvironment, chat_llm: ChatLLM) -> 'SchematicToolAdapter':
        return SchematicToolAdapter(self, template_env, chat_llm)


class ContextualTool(BaseTool):
    @abstractmethod
    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: list[str], tools: dict[str, str] | None = None) -> ToolOutput:
        """Execute the tool with contextual information and yield outputs asynchronously"""
        raise NotImplementedError("ContextualTool subclasses must implement the call method")


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

    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: list[str], tools: dict[str, str] | None = None) -> ToolOutput:
        """Execute the tool with contextual information and return a final ToolOutput"""
        # Check if we need to gather additional context for missing inputs
        missing_fields = await self._identify_missing_inputs(believes, step.intent, context)
        
        if missing_fields and self.inner.requires_prerequisite_steps:
            # Tool can decompose into steps to gather missing data
            steps = await self._build_data_gathering_steps(missing_fields, tools)
            # If we failed to build steps, we have to work with what we have
            if steps:
                return self.decomposition(
                    steps=steps,
                    rationale=f"Need to gather missing inputs: {', '.join(missing_fields)}"
                )
            await runtime.warning(f"Failed to build steps for missing inputs: {', '.join(missing_fields)}. Proceeding with available context.")
        elif missing_fields:
            # Tool cannot decompose, must work with what we have
            await runtime.warning(f"Missing inputs: {', '.join(missing_fields)}. Proceeding with available context.")
        
        # Build arguments from context using LLM
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tool_args = await self._build_tool_arguments(believes, step.intent, context)
            except Exception as e:
                await runtime.warning(f"Error building tool arguments: {str(e)}")
                tool_args = None
            if tool_args is not None:
                await runtime.debug(f"Built tool arguments: {tool_args}")
                break
            if attempt < max_retries - 1:
                await runtime.warning(f"Failed to build tool arguments (attempt {attempt + 1}/{max_retries}). Retrying...")
        else:
            return self.error(f"Failed to build tool arguments after {max_retries} attempts")
        try:
            # Call the inner schematic tool with the built arguments
            result = await self.inner.call(runtime, **tool_args)
            return result
        except Exception as e:
            return self.error(f"Failed to execute schematic tool {self.inner.name}: {str(e)}")

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


class LLMLogicalOperationTool(LLMToolMixin, ContextualTool):
    """Simple tool for LLM-based logical operations and inference tasks.

    This tool handles tasks that require reasoning, planning, summarization,
    or other logical operations using direct LLM processing without decomposition.
    """

    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str | None:
        return "This tool performs logical operations and inference tasks using LLM capabilities. " \
               "It can handle planning, summarization, analysis, reasoning, and other cognitive tasks " \
               "that require understanding context and making logical connections. " \
               "Use this tool when you need to perform operations that involve reasoning, " \
               "pattern recognition, or complex analysis that other specialized tools cannot handle."

    @property
    def output_description(self) -> str | None:
        return "Returns the result of the logical operation or inference task based on the step description and context."

    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: list[str], tools: dict[str, str] | None = None) -> ToolOutput:
        """Execute logical operation based on the step description and context."""
        if believes is None:
            believes = []
        if context is None:
            context = []

        await runtime.debug(f"Performing logical operation: {step.intent}")

        # Perform the logical operation directly using LLM
        response_json = await self._perform_logical_operation(step.intent, context, believes)

        # Parse the JSON response
        try:
            response_data = json.loads(response_json)
        except json.JSONDecodeError as e:
            return self.error(f"Failed to parse JSON response from logical operation: {str(e)}")

        # Check if the operation encountered an error
        if "error_message" in response_data and response_data["error_message"]:
            return self.error(response_data["error_message"])

        # Extract result and rationale
        result = response_data.get("result", "")
        rationale = response_data.get("rationale", "")

        if rationale:
            await runtime.message(f"Logical operation rationale: {rationale}")

        await runtime.message("Completed logical operation successfully")
        return self.result(result)

    async def _perform_logical_operation(self, step_description: str, context: list[str], believes: list[str]) -> str:
        """Perform the logical operation and return the JSON response."""
        return await self.call_template(
            'perform_logical_operation.jinja2',
            json_format=True,
            step_description=step_description,
            context=context,
            believes=believes
        )
