from dataclasses import dataclass

from novelrag.agenturn.pursuit import PursuitOutcome
from novelrag.cli.command import Command
from novelrag.cli.handler.interaction import UndoRedoDetails


@dataclass
class HandlerResult:
    message: list[str] | None = None
    quit: bool = False
    redirect: Command | None = None
    details: PursuitOutcome | UndoRedoDetails | None = None
