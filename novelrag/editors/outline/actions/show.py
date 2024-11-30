from novelrag.core.action import Action, ActionResult
from novelrag.core.exceptions import InvalidMessageFormatError, ActionNotSupportedError
from ..navigation import EventLocation
from ..definitions import OutlineActionConfig
from novelrag.model import Outline
from ..registry import outline_registry


@outline_registry.register('show')
class ShowAction(Action):
    def __init__(self, outline: Outline, current_location: EventLocation, **_kwargs):
        super().__init__()
        self.outline = outline
        self.current_location = current_location
        self.current_event = self.current_location.get_current_event(outline)

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig) -> ('ShowAction', str | None):
        if input_msg:
            raise InvalidMessageFormatError('show', 'outline', input_msg, "show")
        return cls(**config), None

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            raise ActionNotSupportedError('list', 'outline', 'message handling')

        return ActionResult.quit(str(self.current_event))
