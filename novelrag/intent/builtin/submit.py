from novelrag.intent import Intent
from novelrag.intent.action import Action
from novelrag.intent.intent import IntentContext


class Submit(Intent):
    @property
    def default_name(self):
        return 'submit'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        return Action(
            submit=True,
        )
