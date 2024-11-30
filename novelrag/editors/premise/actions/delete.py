import logging
from novelrag.core.action import Action, ActionResult
from novelrag.core.operation import OperationType
from novelrag.editors.premise.definitions import PremiseDefinition, PremiseActionConfig
from novelrag.editors.premise.registry import premise_registry
from novelrag.core.exceptions import InvalidIndexError, InvalidMessageFormatError, ActionNotSupportedError

logger = logging.getLogger(__name__)


@premise_registry.register('delete')
class DeleteAction(Action):
    def __init__(self, idx: int, premises: list[str]):
        super().__init__()
        self.idx = idx
        self.premises = premises
        self.definition = PremiseDefinition()

    @classmethod
    async def create(cls, input_msg: str, **config: PremiseActionConfig):
        try:
            parts = input_msg.split(maxsplit=1)
            idx = int(parts[0])
            if idx < 0 or idx >= len(config['premises']):
                raise InvalidIndexError(idx, len(config['premises']) - 1, "premise")
            if len(parts) > 1:
                raise InvalidMessageFormatError('delete', 'premise', input_msg, "delete INDEX")
            return cls(idx=idx, premises=config['premises']), None
        except ValueError as e:
            raise InvalidMessageFormatError('delete', 'premise', input_msg, "delete INDEX") from e

    async def handle(self, message: str | None) -> ActionResult:
        if message:
            raise ActionNotSupportedError('delete', 'premise', 'message handling')
        return ActionResult.operation(OperationType.DELETE, f'premises.{self.idx}', None)
