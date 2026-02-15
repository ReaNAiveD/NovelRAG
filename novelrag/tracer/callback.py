import logging
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from novelrag.tracer.context import get_current_span
from novelrag.tracer.span import Span, SpanKind

logger = logging.getLogger(__name__)


def _serialize_messages(messages: list[list[BaseMessage]]) -> list[dict[str, Any]]:
    """Convert LangChain message objects into plain dicts for storage.

    In addition to the base ``role`` / ``content`` fields this now captures:

    * **AIMessage.tool_calls** – the tool invocations chosen by the model.
    * **ToolMessage** metadata – ``tool_call_id`` and ``name`` so that
      tool-result messages can be correlated back to the originating call.
    """
    result: list[dict[str, Any]] = []
    for batch in messages:
        for msg in batch:
            entry: dict[str, Any] = {
                "role": msg.type,
                "content": str(msg.content),
            }

            # AIMessage may carry tool-call decisions.
            if isinstance(msg, AIMessage) and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}
                    for tc in msg.tool_calls
                ]

            # ToolMessage carries the result for a specific tool call.
            if isinstance(msg, ToolMessage):
                entry["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    entry["name"] = msg.name

            result.append(entry)
    return result


def _simplify_tool_defs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the essential fields from OpenAI-format tool definitions."""
    simplified: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function", {})
        simplified.append({
            "name": func.get("name", "unknown"),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return simplified


class TracerCallbackHandler(AsyncCallbackHandler):
    """Captures LLM request/response data and attaches it to the active span.

    The handler correlates ``on_chat_model_start`` / ``on_llm_end`` /
    ``on_llm_error`` callbacks via the LangChain *run_id*.

    If no ``LLM_CALL`` span is active when a callback fires the event is
    silently ignored — this keeps the handler safe for use even when tracing
    decorators have not been applied.
    """

    def __init__(self) -> None:
        super().__init__()
        # Map run_id → span so we can correlate start/end events.
        self._run_spans: dict[UUID, Span] = {}

    # ------------------------------------------------------------------
    # LangChain callback interface
    # ------------------------------------------------------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        span = get_current_span()
        if span is None or span.kind != SpanKind.LLM_CALL:
            return

        self._run_spans[run_id] = span

        # Record request details on the span.
        model_name = serialized.get("kwargs", {}).get("model_name") or serialized.get("id", ["unknown"])[-1]
        span.set_attribute("model", model_name)

        request_data: dict[str, Any] = {
            "messages": _serialize_messages(messages),
        }

        # Capture tool definitions when the model was invoked with bind_tools().
        invocation_params = kwargs.get("invocation_params") or {}
        raw_tools: list[dict[str, Any]] = invocation_params.get("tools") or []
        if raw_tools:
            request_data["tools"] = _simplify_tool_defs(raw_tools)

        span.set_attribute("request", request_data)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._run_spans.pop(run_id, None)
        if span is None:
            return

        # Extract response content and tool calls.
        content: str = ""
        msg = None

        if response.generations:
            gen = response.generations[0]
            if gen:
                content = gen[0].text or ""
                # For chat models the message attribute is more reliable.
                msg = getattr(gen[0], "message", None)
                if msg is not None:
                    content = str(msg.content)

        response_data: dict[str, Any] = {"content": content}

        # Capture tool calls returned by the model.
        if isinstance(msg, AIMessage) and msg.tool_calls:
            response_data["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}
                for tc in msg.tool_calls
            ]

        span.set_attribute("response", response_data)

        # Token usage --------------------------------------------------
        # Try the classic llm_output path first, then fall back to the
        # newer ``usage_metadata`` on the message itself.
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or {}
        if not token_usage and msg is not None:
            usage = getattr(msg, "usage_metadata", None)
            if usage is not None:
                token_usage = {
                    "prompt_tokens": getattr(usage, "input_tokens", None),
                    "completion_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
        if token_usage:
            span.set_attribute("token_usage", {
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            })

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._run_spans.pop(run_id, None)
        if span is None:
            return
        span.status = "error"
        span.error = str(error)
