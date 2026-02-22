"""Interaction history for tracking user requests and handler outcomes.

Provides a lightweight, fixed-window history of session interactions that
can be rendered into LLM prompts for context-aware goal translation,
goal deciding, and pursuit assessment.
"""

from dataclasses import dataclass, field
from typing import Literal

from novelrag.agenturn.pursuit import PursuitOutcome


# ---------------------------------------------------------------------------
# Handler details
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UndoRedoDetails:
    """Details for an undo or redo handler invocation."""
    action: Literal["undo", "redo"]
    methods: list[str]
    count: int
    descriptions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        verb = "Undid" if self.action == "undo" else "Redid"
        if self.count == 1:
            header = f"{verb} action: {self.methods[0]}"
        else:
            header = f"{verb} {self.count} actions: {', '.join(self.methods)}"
        if not self.descriptions:
            return header
        lines = [header]
        for i, desc in enumerate(self.descriptions, 1):
            lines.append(f"  {i}. {desc}")
        return "\n".join(lines)


# Union of all possible detail types.
HandlerDetails = PursuitOutcome | UndoRedoDetails


# ---------------------------------------------------------------------------
# Interaction record & history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionRecord:
    """A single request/response interaction within a session."""
    request: str
    handler: str | None = None
    details: PursuitOutcome | UndoRedoDetails | None = None
    message: list[str] | None = None

    def summary(self) -> str:
        """One-line summary suitable for prompt rendering."""
        parts: list[str] = []
        handler_label = self.handler or "agent"
        parts.append(f"[{handler_label}] User: {self.request}")

        if self.details is not None:
            if isinstance(self.details, PursuitOutcome):
                outcome = self.details
                parts.append(
                    f"  → Goal: {outcome.goal.description} | "
                    f"Status: {outcome.status.value} | "
                    f"Steps: {len(outcome.executed_steps)}"
                )
                if outcome.status.value != "completed" and outcome.reason:
                    parts.append(f"  → Reason: {outcome.reason}")
                # Show each executed step with tool name, reason, and result
                if outcome.executed_steps:
                    parts.append("  Steps:")
                    for i, step in enumerate(outcome.executed_steps, 1):
                        tool_name = step.operation.tool or "N/A"
                        status_symbol = "✓" if step.status.name == "SUCCESS" else "✗"
                        reason = step.operation.reason
                        parts.append(f"    {i}. {status_symbol} [{tool_name}] {reason}")
                        result = step.result or step.error_message or "No result"
                        if len(result) > 300:
                            result = result[:300] + "…"
                        parts.append(f"       Result: {result}")
                if outcome.response:
                    resp = outcome.response
                    if len(resp) > 500:
                        resp = resp[:500] + "…"
                    parts.append(f"  → Response: {resp}")
            elif isinstance(self.details, UndoRedoDetails):
                undo_info = self.details
                verb = "Undid" if undo_info.action == "undo" else "Redid"
                parts.append(
                    f"  → {verb} {undo_info.count} action(s): "
                    f"{', '.join(undo_info.methods)}"
                )
                if undo_info.descriptions:
                    parts.append("  Operations:")
                    for i, desc in enumerate(undo_info.descriptions, 1):
                        truncated = desc if len(desc) <= 300 else desc[:300] + "…"
                        parts.append(f"    {i}. {truncated}")
        elif self.message:
            text = "\n".join(self.message)
            if len(text) > 300:
                text = text[:300] + "…"
            parts.append(f"  → {text}")

        return "\n".join(parts)


@dataclass
class InteractionHistory:
    """Fixed-window history of session interactions.

    Shared as a single mutable instance across the session so that
    handlers and the executor pipeline always see the latest state.
    """
    _records: list[InteractionRecord] = field(default_factory=list)

    def add(self, record: InteractionRecord) -> None:
        """Append an interaction record."""
        self._records.append(record)

    def recent(self, n: int = 5) -> list[InteractionRecord]:
        """Return the *n* most recent interaction records."""
        return self._records[-n:]

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        # Always truthy — Prevents `history or default` from
        # accidentally replacing an empty-but-shared instance.
        return True

    def format_recent(self, n: int = 5) -> str:
        """Render the most recent *n* interactions as a prompt-ready string."""
        records = self.recent(n)
        if not records:
            return ""
        return "\n\n".join(r.summary() for r in records)
