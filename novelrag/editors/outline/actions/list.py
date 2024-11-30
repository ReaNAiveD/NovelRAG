import logging
from novelrag.core.action import Action, ActionResult
from novelrag.editors.outline.definitions import OutlineActionConfig
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.core.exceptions import InvalidMessageFormatError, ActionNotSupportedError
from novelrag.model.outline import Outline

logger = logging.getLogger(__name__)


@outline_registry.register('list')
class ListAction(Action):
    def __init__(self, outline: Outline, current_location: EventLocation, **_kwargs):
        super().__init__()
        self.outline = outline
        self.current_location = current_location

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig):
        if input_msg:
            raise InvalidMessageFormatError('list', 'outline', input_msg, "list")
        return cls(**config), None

    def _format_events_list(self) -> str:
        events = self.current_location.get_current_events(self.outline)
        current_path = ' / '.join(self.current_location.get_path_names(self.outline))
        
        header = f"Current location: {current_path}\n\nEvents:"
        events_list = "\n".join(f"{i}. {event}" for i, event in enumerate(events))
        
        return f"{header}\n{events_list}"

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            raise ActionNotSupportedError('list', 'outline', 'message handling')
        
        formatted_list = self._format_events_list()
        return ActionResult.quit(formatted_list)
