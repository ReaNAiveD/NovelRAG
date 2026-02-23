"""Core Procedure support: execution context and error handling.

Two related but distinct concepts in this codebase:

* **LLM Call** – hosts a single LLM invocation together with its input/output
  type transformation (e.g. ``LLMActionDecider``, ``LLMContextAnalyzer``).
* **Procedure** – orchestrates one or more LLM calls, handles the logic between
  them, and manages interaction with the environment through an
  ``ExecutionContext``.

A procedure is a concrete class whose ``execute`` method accepts flat
parameters (just like an LLM-call class) plus an ``ExecutionContext`` and
returns a typed output.  On failure it raises ``ProcedureError`` carrying the
list of effects accomplished before the failure point.

``ExecutionContext`` is the single runtime interface shared by procedures,
tools, and the agent loop.  It provides three unified facets:

* **Messaging** (fire-and-forget): ``info``, ``debug``, ``warning``, ``error``
* **Output**: ``output`` — user-facing content
* **Bidirectional** (blocking): ``confirm``, ``request`` — user confirmation /
  free-form input
"""

import logging
from abc import ABC, abstractmethod


class ExecutionContext(ABC):
    """Runtime context provided to procedures and tools for environment
    interaction.

    Three facets:

    * **Messaging** (fire-and-forget): ``info``, ``debug``, ``warning``,
      ``error`` — developer-facing diagnostic output.
    * **Output**: ``output`` — user-facing content produced during execution.
    * **Bidirectional** (blocking): ``confirm``, ``request`` — interactive
      prompts that pause execution until the user responds.

    Different implementations may route messages to a CLI, a UI, a log file,
    or a test harness.
    """

    # -- Messaging (fire-and-forget) ----------------------------------------

    @abstractmethod
    async def debug(self, content: str):
        """Emit a developer-focused debug message."""
        ...

    @abstractmethod
    async def info(self, content: str):
        """Emit an informational message."""
        ...

    @abstractmethod
    async def warning(self, content: str):
        """Emit a warning about a recoverable condition."""
        ...

    @abstractmethod
    async def error(self, content: str):
        """Emit an error message describing a failure."""
        ...

    # -- Output -------------------------------------------------------------

    @abstractmethod
    async def output(self, content: str):
        """Emit user-facing output produced during execution."""
        ...

    # -- Bidirectional (blocking) -------------------------------------------

    @abstractmethod
    async def confirm(self, prompt: str) -> bool:
        """Request an explicit yes/no confirmation from the user."""
        ...

    @abstractmethod
    async def request(self, prompt: str) -> str:
        """Prompt the user for free-form input and return it."""
        ...


class ProcedureError(Exception):
    """Error raised by a procedure, carrying effects accomplished before failure.

    Unlike a plain exception, ``ProcedureError`` preserves partial progress
    information.  The *effects* list describes what side effects were already
    applied before the failure occurred, enabling callers to decide whether to
    roll back, retry, or propagate.
    """

    def __init__(self, message: str, effects: list[str] | None = None):
        self.message = message
        self.effects: list[str] = effects or []
        super().__init__(message)
