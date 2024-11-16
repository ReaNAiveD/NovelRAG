from novelrag.action import Action
from novelrag.aspect import AspectContext, AspectRegistry
from novelrag.editors.premise.actions import UpdateAction, ListAction, DeleteAction
from novelrag.editors.premise.actions.create import CreateAction
from novelrag.editors.premise.actions.default import DefaultAction
from novelrag.editors.premise.definitions import PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry

class PremiseAspectContext(AspectContext):
    registry = premise_registry  # Set the registry for this aspect
    
    def __init__(self, file_path: str, oai_config: dict, chat_params: dict):
        super().__init__('premise', file_path)
        self.file_path = file_path
        self.action_config = PremiseActionConfig(
            premises=self.load_file()['premises'],
            oai_config=oai_config,
            chat_params=chat_params
        )

    async def act(self, action_name: str | None, msg: str | None) -> tuple[Action, str | None]:
        if action_name is None:
            return await DefaultAction.create(msg, **self.action_config)
            
        match action_name:
            case 'update':
                return await UpdateAction.create(msg, **self.action_config)
            case 'list':
                return await ListAction.create(msg, **self.action_config)
            case 'delete':
                return await DeleteAction.create(msg, **self.action_config)
            case 'create':
                return await CreateAction.create(msg, **self.action_config)
        return await super().act(action_name, msg)


def build_context(config: dict):
    return PremiseAspectContext(**config)
