from abc import ABC, abstractmethod
import json
import logging
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment
from .steps import StepDefinition

from .types import ToolOutput, ToolResult, ToolError

logger = logging.getLogger(__name__)


class LLMToolMixin:
    """Mixin that provides LLM template calling functionality"""
    
    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.template_env = template_env
        self.chat_llm = chat_llm
    
    async def call_template(self, template_name: str, user_question: str | None = None, json_format: bool = False, **kwargs: bool | int | float | str | list | dict) -> str:
        """Call an LLM with a template and return the response."""
        logger.info(f"Calling template: {template_name} with json_format={json_format} ─────────────────")
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        logger.debug('\n' + prompt)
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        response = await self.chat_llm.chat(messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please answer the question.'}
        ], response_format='json_object' if json_format else None)
        logger.info(f"Received response from LLM for template {template_name}")
        logger.info('\n' + response) 
        logger.info('───────────────────────────────────────────────────────────────────────────────')
        return response


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


class SchematicTool(BaseTool):

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
    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: dict[str, list[str]], tools: dict[str, str] | None = None) -> ToolOutput:
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

    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: dict[str, list[str]], tools: dict[str, str] | None = None) -> ToolOutput:
        """Execute the tool with contextual information and return a final ToolOutput"""
        # Prepare execution by checking breakdown needs and building tool arguments in one step
        analysis_result = await self._prepare_execution(believes, step, context, tools)
        
        if analysis_result.get("requires_breakdown", False):
            # Tool needs breakdown - return the ToolError with breakdown data
            breakdown_info = analysis_result["breakdown"]
            await runtime.error(f"Step requires breakdown: {breakdown_info.get('reason', 'Breakdown required')}")
            
            # Create structured error message with breakdown data
            error_info = {
                "is_decomposition": True,
                "rationale": breakdown_info.get("reason", "Breakdown required"),
                "rerun": breakdown_info.get("rerun", False),
                "available_context": breakdown_info.get("available_context", ""),
                "missing_requirements": breakdown_info.get("missing_requirements", []),
                "blocking_conditions": breakdown_info.get("blocking_conditions", [])
            }
            error_message = f"DECOMPOSITION_REQUIRED: {json.dumps(error_info, ensure_ascii=False)}"
            return self.error(error_message)
        
        # Extract tool arguments from analysis result
        tool_args = analysis_result.get("tool_arguments", {})
        if not tool_args:
            return self.error("Failed to build tool arguments from analysis")
            
        try:
            # Call the inner schematic tool with the built arguments
            await runtime.message(f"Executing tool '{self.inner.name}' with arguments: {tool_args}")
            await runtime.debug(f"Built tool arguments: {tool_args}")
            result = await self.inner.call(runtime, **tool_args)
            return result
        except Exception as e:
            return self.error(f"Failed to execute schematic tool {self.inner.name}: {str(e)}")

    async def _prepare_execution(self, believes: list[str], step: StepDefinition, context: dict[str, list[str]], tools: dict[str, str] | None = None) -> dict[str, Any]:
        """Prepare tool execution by determining if breakdown is needed and building arguments in a single LLM call"""
        try:
            analysis_json = await self.call_template(
                "schematic_tool_preparation.jinja2",
                tool_name=self.inner.name,
                tool_description=self.inner.description or "No description available",
                output_description=self.inner.output_description or "",
                step_description=step.intent,
                context=context or {},
                believes=believes or [],
                input_schema=self.inner.input_schema,
                available_tools=tools or {},
                json_format=True
            )

            # Parse the JSON response
            analysis_data = json.loads(analysis_json)
            return analysis_data
            
        except (json.JSONDecodeError, Exception) as e:
            # Fall back to error case on any parsing issues
            return {
                "requires_breakdown": True,
                "breakdown": {
                    "reason": f"Failed to prepare step: {str(e)}",
                    "available_context": "Preparation failed due to parsing error",
                    "missing_requirements": ["Valid preparation response"],
                    "blocking_conditions": ["LLM response parsing failure"],
                    "rerun": True
                }
            }




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

    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: dict[str, list[str]], tools: dict[str, str] | None = None) -> ToolOutput:
        """Execute logical operation based on the step description and context."""
        if believes is None:
            believes = []
        if context is None:
            context = {}

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

    async def _perform_logical_operation(self, step_description: str, context: dict[str, list[str]], believes: list[str]) -> str:
        """Perform the logical operation and return the JSON response."""
        return await self.call_template(
            'perform_logical_operation.jinja2',
            json_format=True,
            step_description=step_description,
            context=context,
            believes=believes
        )
