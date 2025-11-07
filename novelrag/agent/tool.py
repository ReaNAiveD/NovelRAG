from abc import ABC, abstractmethod
import json
import logging
import time
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment
from .steps import StepDefinition
from .llm_logger import LLMRequest, LLMResponse, log_llm_call

from .types import ToolOutput, ToolResult, ToolError

logger = logging.getLogger(__name__)


class LLMToolMixin:
    """Mixin that provides LLM template calling functionality"""
    
    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.template_env = template_env
        self.chat_llm = chat_llm

    async def call_template(self, template_name: str, user_question: str | None = None, json_format: bool = False, **kwargs: None | bool | int | float | str | list | dict) -> str:
        """Call an LLM with a template and return the response."""
        logger.info(f"Calling template: {template_name} with json_format={json_format} ─────────────────")
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        logger.debug('\n' + prompt)
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        
        # Prepare request for logging
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please answer the question.'}
        ]
        request = LLMRequest(
            messages=messages,
            response_format='json_object' if json_format else None
        )
        
        # Call LLM and measure time
        start_time = time.time()
        response = await self.chat_llm.chat(
            messages=messages,
            response_format='json_object' if json_format else None
        )
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log the call
        log_response = LLMResponse(content=response)
        log_llm_call(template_name, request, log_response, duration_ms)
        
        logger.info(f"Received response from LLM for template {template_name}")
        logger.info('\n' + response) 
        logger.info('───────────────────────────────────────────────────────────────────────────────')
        return response

    async def call_template_structured(self, template_name: str, response_schema: dict[str, Any], user_question: str | None = None, **kwargs: None | bool | int | float | str | list | dict) -> str:
        """Call an LLM with a template and enforce a specific JSON schema for the response.
        
        Args:
            template_name: Name of the template to load and render
            response_schema: JSON schema definition that the response must conform to
            user_question: Optional user question to include in the chat
            **kwargs: Template variables to pass to the template renderer
            
        Returns:
            The LLM response as a string (should be valid JSON conforming to the schema)
        """
        logger.info(f"Calling template: {template_name} with structured schema ─────────────────")
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        logger.debug('\n' + prompt)
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        logger.debug(f"Response schema: {json.dumps(response_schema, indent=2)}")
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        
        # Prepare request for logging
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please provide a response that conforms to the specified JSON schema.'}
        ]
        response_format_config = {
            "type": "json_schema", 
            "json_schema": {
                "name": "structured_response",
                "schema": response_schema,
                "strict": True
            }
        }
        request = LLMRequest(
            messages=messages,
            response_format=json.dumps(response_format_config)  # Store as JSON string for logging
        )
        
        # Call LLM and measure time
        start_time = time.time()
        response = await self.chat_llm.chat(
            messages=messages,
            response_format=response_format_config
        )
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log the call
        log_response = LLMResponse(content=response)
        log_llm_call(template_name, request, log_response, duration_ms)
        
        logger.info(f"Received structured response from LLM for template {template_name}")
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


class SchematicTool(BaseTool):

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Schema for input parameters required by the tool"""
        pass

    @property
    def require_context(self) -> bool:
        """Whether the tool requires context to operate effectively"""
        return False

    @abstractmethod
    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Execute the tool with provided parameters and yield outputs asynchronously"""
        raise NotImplementedError("SchematicTool subclasses must implement the call method")
