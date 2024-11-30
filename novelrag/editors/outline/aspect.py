import logging

from novelrag.core.aspect import AspectContext
from novelrag.core.storage import NovelStorage
from novelrag.editors.outline.actions import *
from novelrag.editors.outline.registry import outline_registry
from novelrag.editors.outline.definitions import OutlineActionConfig
from novelrag.editors.outline.navigation import EventLocation
from novelrag.model.outline import Outline

logger = logging.getLogger(__name__)


class OutlineAspectContext(AspectContext):
    registry = outline_registry
    
    def __init__(self, storage: NovelStorage, oai_config: dict, chat_params: dict):
        super().__init__('outline', storage['outline'])
        self.current_location = EventLocation([])  # Start at root
        self.storage = storage
        self.oai_config = oai_config
        self.chat_params = chat_params

    @property
    def _outline(self) -> Outline:
        return self.storage['outline'].data
    
    @property
    def _path_names(self):
        return self.current_location.get_path_names(self._outline)

    @property
    def prompt_section(self):
        try:
            path_names = self._path_names
        except ValueError as e:
            logger.error(f"Error getting path names: {e}\nResetting to root")
            self.current_location = EventLocation([])
            path_names = []
        if not path_names:
            return self.name
        return f"{self.name} > {' > '.join(path_names)}\n  "

    @property
    def action_config(self):
        return OutlineActionConfig(
            outline=self._outline,
            premise=self.storage.premise().data,
            current_location=self.current_location,
            oai_config=self.oai_config,
            chat_params=self.chat_params
        )


def build_context(storage: NovelStorage, config: dict):
    return OutlineAspectContext(storage, **config)
