from novelrag.intent import Intent
from novelrag.intent.action import Action
from novelrag.intent.intent import IntentContext


class Undo(Intent):
    @property
    def default_name(self):
        return 'undo'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        return Action(
            undo=True,
        )
