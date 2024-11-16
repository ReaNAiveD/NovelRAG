import logging
from novelrag.action import Action, ActionResult
from novelrag.aspect_editors.premise.definitions import PremiseDefinition, PremiseActionConfig

logger = logging.getLogger(__name__)


class ListAction(Action):
    def __init__(self, premises: list[str]):
        super().__init__()
        self.premises = premises
        self.definition = PremiseDefinition()

    @property
    def name(self):
        return 'list'

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        if input_msg:
            logger.warning('Action "list" doesn\'t accept a message as input.')
        return cls(premises=config['premises']), None

    async def handle(self, message: str | None) -> ActionResult:
        formatted_list = self.definition.format_premises_list(self.premises)
        return ActionResult.quit(formatted_list)
