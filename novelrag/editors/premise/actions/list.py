import logging
from novelrag.core.action import Action, ActionResult
from novelrag.editors.premise.definitions import PremiseDefinition, PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry
from novelrag.core.exceptions import InvalidMessageFormatError, ActionNotSupportedError

logger = logging.getLogger(__name__)


@premise_registry.register('list')
class ListAction(Action):
    def __init__(self, premises: list[str]):
        super().__init__()
        self.premises = premises
        self.definition = PremiseDefinition()

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        if input_msg:
            raise InvalidMessageFormatError('list', 'premise', input_msg, "list")
        return cls(premises=config['premises']), None

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            raise ActionNotSupportedError('list', 'premise', 'message handling')
        formatted_list = self.definition.format_premises_list(self.premises)
        return ActionResult.quit(formatted_list)
