from novelrag.core.aspect import AspectContext
from novelrag.core.storage import NovelStorage
from novelrag.editors.premise.actions import *
from novelrag.editors.premise.definitions import PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry


class PremiseAspectContext(AspectContext):
    registry = premise_registry  # Set the registry for this aspect
    
    def __init__(self, storage: NovelStorage, oai_config: dict, chat_params: dict):
        super().__init__('premise', storage.premise())
        self.oai_config = oai_config
        self.chat_params = chat_params
        self.storage = storage

    @property
    def action_config(self):
        return PremiseActionConfig(
            premises=self.storage.premise().data.premises,
            oai_config=self.oai_config,
            chat_params=self.chat_params
        )


def build_context(storage: NovelStorage, config: dict):
    return PremiseAspectContext(storage, **config)
