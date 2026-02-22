"""Runtime interaction interface for tools.

This module originally provided the abstract ``ToolRuntime`` interface.
``ToolRuntime`` is now a **deprecated alias** for
:class:`~novelrag.agenturn.procedure.ExecutionContext`, which serves as the
single runtime context for procedures, tools, and the agent loop.

Existing code that imports or subclasses ``ToolRuntime`` will continue to
work.
"""

from novelrag.agenturn.procedure import ExecutionContext

# Backward-compatible alias — new code should use ExecutionContext directly.
ToolRuntime = ExecutionContext
