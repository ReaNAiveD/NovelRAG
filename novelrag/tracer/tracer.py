import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from typing import Any, AsyncIterator, Optional

from novelrag.tracer.callback import TracerCallbackHandler
from novelrag.tracer.exporter import YAMLExporter
from novelrag.tracer.span import Span, SpanKind, _current_span, get_current_span, set_current_span

logger = logging.getLogger(__name__)


class Tracer:
    """Hierarchical span-based tracer.

    Parameters
    ----------
    exporter:
        An exporter used to persist the trace tree when :pymethod:`export`
        is called.  May be ``None`` (trace data is kept only in memory).
    """

    def __init__(
        self,
        exporter: YAMLExporter | None = None,
    ) -> None:
        self._exporter = exporter
        self._callback_handler = TracerCallbackHandler()
        self._session_span: Optional[Span] = None

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    def activate(self) -> Token:
        """Push this tracer into the ``ContextVar`` so decorators find it."""
        return set_active_tracer(self)

    def deactivate(self, token: Token) -> None:
        """Restore the previous tracer (or ``None``) via *token*."""
        _active_tracer.reset(token)

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[Span, Token]:
        """Create a new span and make it the *current* span.

        The new span is automatically added as a child of the currently
        active span (if any).

        Returns ``(span, context_token)`` — the token must be passed to
        :pymethod:`end_span` to restore the previous span.
        """
        span = Span(kind=kind, name=name)
        if attributes:
            span.attributes.update(attributes)

        parent = get_current_span()
        if parent is not None:
            parent.add_child(span)

        if kind == SpanKind.SESSION:
            self._session_span = span

        token = set_current_span(span)
        return span, token

    def end_span(
        self,
        span: Span,
        token: Token,
        error: Exception | None = None,
    ) -> None:
        """Finish *span* and restore the previous span via *token*."""
        span.finish(error=error)
        _current_span.reset(token)

    # ------------------------------------------------------------------
    # Convenience: async context manager for inline LLM spans
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def llm_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Context-manager that wraps a block in an ``LLM_CALL`` span.

        Usage::

            async with tracer.llm_span("exploration_goal"):
                response = await llm.ainvoke(...)
        """
        span, token = self.start_span(SpanKind.LLM_CALL, name, attributes)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = str(exc)
            raise
        finally:
            self.end_span(span, token)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> None:
        """Persist the session trace tree via the configured exporter."""
        if self._exporter is None:
            logger.debug("No exporter configured — skipping trace export.")
            return
        if self._session_span is None:
            logger.warning("No session span recorded — nothing to export.")
            return
        self._exporter.export(self._session_span)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def callback_handler(self) -> TracerCallbackHandler:
        """The LangChain callback handler managed by this tracer."""
        return self._callback_handler

    @property
    def session_span(self) -> Optional[Span]:
        """The root session span (set by ``@trace_session``)."""
        return self._session_span


# ---------------------------------------------------------------------------
# Active tracer ContextVar
# ---------------------------------------------------------------------------
# Defined after the ``Tracer`` class so the generic parameter can use the
# real type instead of ``Any``.  Method bodies in ``Tracer`` reference these
# names, which Python resolves at *call* time (not at class-definition time).

_active_tracer: ContextVar[Tracer | None] = ContextVar(
    "active_tracer", default=None,
)


def get_active_tracer() -> Tracer | None:
    """Return the tracer that was most recently activated, or ``None``."""
    return _active_tracer.get()


def set_active_tracer(tracer: Tracer | None) -> Token:
    """Push *tracer* into the context, returning a reset token."""
    return _active_tracer.set(tracer)
