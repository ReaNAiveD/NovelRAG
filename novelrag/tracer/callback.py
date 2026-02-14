import logging
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from novelrag.tracer.context import get_current_span
from novelrag.tracer.span import Span, SpanKind

logger = logging.getLogger(__name__)


def _serialize_messages(messages: list[list[BaseMessage]]) -> list[dict[str, str]]:
    """Convert LangChain message objects into plain dicts for storage."""
    result: list[dict[str, str]] = []
    for batch in messages:
        for msg in batch:
            result.append({
                "role": msg.type,
                "content": str(msg.content),
            })
    return result


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
        span.set_attribute("request", {
            "messages": _serialize_messages(messages),
        })

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

        # Extract response content.
        content: str = ""
        if response.generations:
            gen = response.generations[0]
            if gen:
                content = gen[0].text or ""
                # For chat models the message attribute is more reliable.
                msg = getattr(gen[0], "message", None)
                if msg is not None:
                    content = str(msg.content)

        span.set_attribute("response", {"content": content})

        # Token usage --------------------------------------------------
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or {}
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
