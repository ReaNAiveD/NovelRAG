from novelrag.cli.command import Command
from novelrag.cli.handler.handler import Handler
from novelrag.cli.handler.result import HandlerResult


class QuitHandler(Handler):
    async def handle(self, command: Command) -> HandlerResult:
        return HandlerResult(
            quit=True,
        )
