"""Tracer decorators for the five span levels.

Each decorator creates a span of the appropriate :class:`SpanKind`, pushes
it as the *current* span for the duration of the decorated ``async`` call,
and pops it on exit.  If no tracer has been :pymethod:`activate`-d the
decorated function runs untraced (zero overhead).

Usage::

    from novelrag.tracer import trace_session, trace_intent, trace_pursuit, trace_tool, trace_llm

    @trace_session("shell_session")
    async def run(self):
        ...

    @trace_intent("handle_request")
    async def handle_request(self, request: str) -> str:
        ...

    @trace_pursuit("handle_goal")
    async def handle_goal(self, goal: Goal) -> PursuitOutcome:
        ...

    @trace_tool()                          # name auto-extracted from args
    async def _execute_tool(self, tool_name, ...):
        ...

    @trace_llm("goal_translation")
    async def translate(self, ...):
        ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from novelrag.tracer.context import get_active_tracer
from novelrag.tracer.span import SpanKind

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_decorator(
    kind: SpanKind,
    name: str | None = None,
    *,
    auto_export: bool = False,
    name_kwarg: str | None = None,
) -> Callable[[F], F]:
    """Build a decorator that wraps an *async* function in a span.

    Parameters
    ----------
    kind:
        The semantic span level.
    name:
        Fixed label for the span.  If ``None`` the function name is used,
        or a keyword argument can be inspected (see *name_kwarg*).
    auto_export:
        If ``True`` the tracer's ``export()`` method is called after the
        span finishes (used by ``@trace_session``).
    name_kwarg:
        If set, the span name is read from the keyword argument with this
        name at call time.  Useful for ``@trace_tool()`` where the tool
        name comes from the ``tool_name`` parameter.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_active_tracer()
            if tracer is None:
                # No tracer active — run the function untraced.
                return await fn(*args, **kwargs)

            # Determine span name.
            span_name = name or fn.__name__
            if name_kwarg is not None:
                span_name = kwargs.get(name_kwarg) or span_name
                # Also try positional args via parameter inspection.
                if span_name == fn.__name__:
                    import inspect
                    sig = inspect.signature(fn)
                    params = list(sig.parameters.keys())
                    if name_kwarg in params:
                        idx = params.index(name_kwarg)
                        if idx < len(args):
                            span_name = str(args[idx])

            span, token = tracer.start_span(kind, span_name)
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception as exc:
                span.status = "error"
                span.error = str(exc)
                raise
            finally:
                tracer.end_span(span, token)
                if auto_export:
                    tracer.export()

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Public decorators
# ---------------------------------------------------------------------------


def trace_session(name: str | None = None) -> Callable[[F], F]:
    """Mark an async function as a **session**-level span.

    The session span is the root of the trace tree.  When it ends the
    tracer automatically exports the collected data.
    """
    return _make_decorator(SpanKind.SESSION, name, auto_export=True)


def trace_intent(name: str | None = None) -> Callable[[F], F]:
    """Mark an async function as an **intent**-level span.

    One intent corresponds to a single user command or autonomous goal
    decision.
    """
    return _make_decorator(SpanKind.INTENT, name, auto_export=True)


def trace_pursuit(name: str | None = None) -> Callable[[F], F]:
    """Mark an async function as a **pursuit**-level span.

    A pursuit is the execution of a single goal.
    """
    return _make_decorator(SpanKind.PURSUIT, name)


def trace_tool(
    name: str | None = None,
    *,
    name_kwarg: str = "tool_name",
) -> Callable[[F], F]:
    """Mark an async function as a **tool-call**-level span.

    By default the span name is read from the ``tool_name`` keyword
    argument of the decorated function.
    """
    return _make_decorator(SpanKind.TOOL_CALL, name, name_kwarg=name_kwarg)


def trace_llm(name: str) -> Callable[[F], F]:
    """Mark an async function as an **LLM-call**-level span.

    The :class:`~novelrag.tracer.callback.TracerCallbackHandler` will
    automatically attach LLM request/response data to this span when the
    underlying ``ainvoke`` call fires LangChain callbacks.
    """
    return _make_decorator(SpanKind.LLM_CALL, name)
