import abc

from novelrag.operation import OperationType, Operation


class ActionResult:
    def __init__(self, result_type: str):
        self.type = result_type

    @classmethod
    def message(cls, message: str):
        return MessageResult(message)

    @classmethod
    def operation(cls, op_type: OperationType, path: str, data):
        return OperationResult(Operation(
            type=op_type, path=path, data=data
        ))

    @classmethod
    def quit(cls, message: str | None = None):
        return QuitResult(message)


class MessageResult(ActionResult):
    def __init__(self, message: str):
        super().__init__('message')
        self.message = message


class OperationResult(ActionResult):
    def __init__(self, op: Operation):
        super().__init__('op')
        self.op = op


class QuitResult(ActionResult):
    def __init__(self, message: str | None = None):
        super().__init__('quit')
        self.message = message


class Action(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    async def handle(self, message: str | None) -> ActionResult:
        pass

    async def handle_command(self, command: str, message: str | None) -> ActionResult:
        match command:
            case 'quit' | 'q':
                return await self.quit(message)
            case _:
                raise Exception("Unrecognized Command")

    async def quit(self, message: str | None) -> ActionResult:
        return ActionResult.quit()
