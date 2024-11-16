import logging
from novelrag.action import Action, ActionResult
from novelrag.operation import OperationType
from novelrag.editors.premise.definitions import PremiseDefinition, PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry

logger = logging.getLogger(__name__)


@premise_registry.register('delete')
class DeleteAction(Action):
    def __init__(self, idx: int, premises: list[str]):
        super().__init__()
        self.idx = idx
        self.premises = premises
        self.definition = PremiseDefinition()

    @property
    def name(self):
        return 'delete'

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        parts = input_msg.split(maxsplit=1)
        idx = int(parts[0])
        if len(parts) > 1:
            logger.warning('Action "delete" doesn\'t accept additional message input.')
        return cls(idx=idx, premises=config['premises']), None

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            logger.warning('Delete action does not process additional messages.')
        return ActionResult.operation(OperationType.DELETE, f'premises.{self.idx}', None)
