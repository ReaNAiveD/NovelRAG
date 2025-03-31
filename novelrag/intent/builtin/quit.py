from novelrag.intent import Intent
from novelrag.intent.action import Action
from novelrag.intent.intent import IntentContext


class Quit(Intent):
    @property
    def default_name(self) -> str | None:
        return 'quit'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        return Action(
            quit=True,
        )
