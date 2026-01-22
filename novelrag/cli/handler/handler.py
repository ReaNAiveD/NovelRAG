from typing import Protocol

from novelrag.cli.command import Command

from .result import HandlerResult


class Handler(Protocol):
    async def handle(self, command: Command) -> HandlerResult:
        ...
