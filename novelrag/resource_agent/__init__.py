"""Resource-agent subsystem.

This package provides resource-specific agent logic built on top of the
generic ``agenturn`` framework.  Use :func:`create_executor` from
:mod:`novelrag.resource_agent.factory` to wire everything together.

Sub-packages provide the implementation details and can be imported
directly when needed:

* ``action_determine`` – multi-phase orchestration loop
* ``backlog`` – priority work queue
* ``goal_decider`` – autonomous goal generation
* ``propose`` – content proposal generation
* ``tool`` – resource tools
* ``workspace`` – dynamic resource context management
"""

from .backlog.types import Backlog, BacklogEntry
from .factory import create_executor
from .undo import ReversibleAction, UndoQueue
from .workspace import (
    ResourceContext,
    ContextWorkspace,
    ContextSnapshot,
    ResourceSegment,
    SegmentData,
    SearchHistoryItem,
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
