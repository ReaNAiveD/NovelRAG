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
"""

import logging
from abc import ABC, abstractmethod


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
