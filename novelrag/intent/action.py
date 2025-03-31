from dataclasses import dataclass

from novelrag.pending_queue import PendingUpdateItem


@dataclass
class UpdateAction:
    item: PendingUpdateItem
    overwrite_pending: bool = True


@dataclass
class Redirect:
    aspect: str | None
    intent: str | None
    message: str | None


@dataclass
class Action:
    cd: str | None = None
    message: list[str] | None = None
    update: UpdateAction | None = None
    submit: bool = False
    undo: bool = False
    quit: bool = False
    redirect: Redirect | None = None

    def verify(self):
        pass
