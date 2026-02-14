import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SpanKind(str, Enum):
    """Hierarchy level of a span inside a trace tree.

    The expected nesting order (outermost → innermost) is::

        SESSION  →  INTENT  →  PURSUIT  →  TOOL_CALL  →  LLM_CALL

    Not every span must contain all levels; any span may directly nest
    any deeper-level span.
    """

    SESSION = "session"
    INTENT = "intent"
    PURSUIT = "pursuit"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


def _short_id() -> str:
    """Return a short, collision-resistant span identifier."""
    return uuid.uuid4().hex[:12]


@dataclass
class Span:
    """A single node in the trace tree.

    Parameters
    ----------
    kind:
        The semantic level of this span.
    name:
        A human-readable label (e.g. ``"goal_translation"``).
    """

    kind: SpanKind
    name: str
    span_id: str = field(default_factory=_short_id)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    error: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list['Span'] = field(default_factory=list)
    parent: Optional['Span'] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an arbitrary key/value pair to this span."""
        self.attributes[key] = value

    def add_child(self, child: 'Span') -> None:
        """Register *child* as a sub-span of this span."""
        child.parent = self
        self.children.append(child)

    def finish(self, error: Exception | None = None) -> None:
        """Mark the span as finished, computing *duration_ms*.

        If *error* is provided the span status is set to ``"error"`` and the
        error message is recorded.
        """
        self.end_time = datetime.now()
        self.duration_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000
        )
        if error is not None:
            self.status = "error"
            self.error = str(error)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Recursively serialize this span (and all children) to a plain dict."""
        d: dict[str, Any] = {
            "kind": self.kind.value,
            "name": self.name,
            "span_id": self.span_id,
            "start_time": self.start_time.isoformat(),
        }
        if self.end_time is not None:
            d["end_time"] = self.end_time.isoformat()
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 2)
        d["status"] = self.status
        if self.error is not None:
            d["error"] = self.error
        if self.attributes:
            d["attributes"] = self.attributes
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d
