import logging
from novelrag.intent import Intent, IntentContext
from novelrag.intent.action import Action

logger = logging.getLogger(__name__)


class List(Intent):

    @property
    def default_name(self) -> str | None:
        return 'list'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        elements = context.current_aspect.root_elements
        return Action(
            message=['* ' + element['cne'] for element in elements],
        )
