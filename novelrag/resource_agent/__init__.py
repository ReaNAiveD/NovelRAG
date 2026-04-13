"""Resource-agent subsystem.

This package provides resource-specific agent logic built on top of the
generic ``agenturn`` framework.  Use :func:`create_executor` from
:mod:`novelrag.resource_agent.factory` to wire everything together.

Sub-packages provide the implementation details and can be imported
directly when needed:

* ``action_determine`` – multi-phase orchestration loop
* ``backlog`` – backlog interface (protocol and data types)
* ``goal_decider`` – autonomous goal generation
* ``propose`` – content proposal generation
* ``tool`` – resource tools
* ``workspace`` – dynamic resource context management
"""

from .backlog import Backlog, BacklogEntry
from .factory import create_executor
from .undo import ReversibleAction, UndoQueue
from .workspace import (
    ContextSnapshot,
    ContextWorkspace,
    ResourceContext,
    ResourceSegment,
    SearchHistoryItem,
    SegmentData,
)

__all__ = [
    # Factory
    "create_executor",
    # Workspace
    "ResourceContext",
    "ContextWorkspace",
    "ContextSnapshot",
    "ResourceSegment",
    "SegmentData",
    "SearchHistoryItem",
    # Undo
    "ReversibleAction",
    "UndoQueue",
    # Backlog
    "Backlog",
    "BacklogEntry",
]
