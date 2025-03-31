from novelrag.intent.intent import Intent, ActionResult
from novelrag.exceptions import InvalidMessageFormatError
from novelrag.editors.outline.definitions import OutlineActionConfig
from novelrag.editors.outline.navigation import EventLocation
from novelrag.editors.outline.registry import outline_registry
from novelrag.model.outline import Outline


@outline_registry.register('cd')
class ChangeDirectoryIntent(Intent):
    def __init__(self, outline: Outline, current_location: EventLocation, **_kwargs):
        super().__init__()
        self.outline = outline
        self.current_location = current_location

    @classmethod
    async def create(cls, input_msg: str, **config: OutlineActionConfig):
        if not input_msg:
            raise InvalidMessageFormatError(
                'cd',
                'path',
                input_msg,
                "cd <path> - where path can be: number, .., or / (root)"
            )
        return cls(**config), input_msg

    @staticmethod
    def _validate_index(idx: int, events: list) -> bool:
        return 0 <= idx < len(events)

    async def handle(self, message: str) -> ActionResult:
        message = message.strip()
        current_events = self.current_location.get_current_events(self.outline)

        # Handle special paths
        if message == '/':  # Root
            self.current_location.path = []
            return ActionResult.quit("Moved to root")
        elif message == '..':  # Parent
            if not self.current_location.path:
                return ActionResult.quit("Already at root")
            self.current_location.path.pop()
            return ActionResult.quit("Moved up one level")

        # Handle numeric index
        try:
            target_idx = int(message)
            if not self._validate_index(target_idx, current_events):
                return ActionResult.quit(
                    f"Invalid index: {target_idx}. Must be between 0 and {len(current_events) - 1}"
                )
                
            self.current_location.path.append(target_idx)
            return ActionResult.quit(f"Moved to event {target_idx}")
            
        except ValueError:
            raise InvalidMessageFormatError(
                'cd',
                'path',
                message,
                "cd <path> - where path can be: number, .., or / (root)"
            )
