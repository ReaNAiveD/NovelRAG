"""Core Procedure abstraction for typed units of execution.

A **Procedure** is a generic, typed unit of execution parameterized by an input
type and an output type.  It has a single async method ``execute`` that receives
the typed input and an ``ExecutionContext``, and returns the typed output on
success.

On failure, a procedure raises a ``ProcedureError`` exception that carries the
list of effects accomplished before the failure point.  This preserves partial
progress information so that callers can inspect what side effects were already
applied.

Two related but distinct concepts in this codebase:

* **LLM Call** – hosts a single LLM invocation together with its input/output
  type transformation (e.g. ``LLMActionDecider``, ``LLMContextAnalyzer``).
* **Procedure** – orchestrates one or more LLM calls, handles the logic between
  them, and manages interaction with the environment through an
  ``ExecutionContext``.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExecutionContext(ABC):
    """Runtime context provided to procedures for environment interaction.

    Procedures use this context to communicate with the external environment
    (logging, messaging, etc.) rather than returning side-effect information in
    their output types.  Different implementations may route messages to a CLI,
    a UI, a log file, or a test harness.
    """

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


class Procedure(ABC, Generic[InputT, OutputT]):
    """Generic typed unit of execution.

    A Procedure receives a typed input and an ``ExecutionContext``, performs its
    work (potentially including LLM calls, environment mutations, and
    sub-procedure invocations), and returns a typed output.

    On failure it raises ``ProcedureError`` with the list of effects
    accomplished before the failure point.
    """

    @abstractmethod
    async def execute(self, procedure_input: InputT, ctx: ExecutionContext) -> OutputT:
        """Execute the procedure.

        Args:
            procedure_input: Typed input for this procedure.
            ctx: Execution context for environment interaction.

        Returns:
            Typed output of the procedure.

        Raises:
            ProcedureError: On failure, with effects accomplished before failure.
        """
        ...


class LoggingExecutionContext(ExecutionContext):
    """``ExecutionContext`` backed by Python's :mod:`logging` module.

    Useful as a default context when no richer runtime (CLI, UI, …) is
    available.  Each message level maps directly to the corresponding
    ``logging`` level.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)

    async def debug(self, content: str):
        self._logger.debug(content)

    async def info(self, content: str):
        self._logger.info(content)

    async def warning(self, content: str):
        self._logger.warning(content)

    async def error(self, content: str):
        self._logger.error(content)
