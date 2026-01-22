from dataclasses import dataclass

from novelrag.cli.command import Command


@dataclass
class HandlerResult:
    message: list[str] | None = None
    quit: bool = False
    redirect: Command | None = None
