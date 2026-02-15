"""Context-variable helpers for the tracer framework."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from novelrag.tracer.span import Span
    from novelrag.tracer.tracer import Tracer

# ---------------------------------------------------------------------------
# ContextVars  (use ``Any`` to avoid evaluating forward refs at runtime)
# ---------------------------------------------------------------------------

_current_span: ContextVar[Any] = ContextVar(
    "current_span", default=None,
)

_active_tracer: ContextVar[Any] = ContextVar(
    "active_tracer", default=None,
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_current_span() -> Optional[Span]:
    """Return the span that is currently active, or ``None``."""
    return _current_span.get()


def set_current_span(span: Optional[Span]) -> Token:
    """Make *span* the active span, returning a reset token."""
    return _current_span.set(span)


def get_active_tracer() -> Optional[Tracer]:
    """Return the tracer that was most recently :pymethod:`activate`-d."""
    return _active_tracer.get()


def set_active_tracer(tracer: Optional[Tracer]) -> Token:
    """Push *tracer* into the context, returning a reset token."""
    return _active_tracer.set(tracer)
