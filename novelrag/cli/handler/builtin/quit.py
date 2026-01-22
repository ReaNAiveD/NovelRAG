from novelrag.cli.handler.result import HandlerResult
from novelrag.cli.handler.handler import Handler
from novelrag.cli.command import Command


class QuitHandler(Handler):
    async def handle(self, command: Command) -> HandlerResult:
        return HandlerResult(
            quit=True,
        )
