import yaml

from novelrag.action import Action, QuitResult, MessageResult, OperationResult
from novelrag.operation import apply_operation


class AspectContext:
    def __init__(self, name: str, file_path: str | None):
        self.name = name
        self.file_path = file_path
        self._aspect_data = None

    @property
    def aspect_data(self):
        if self._aspect_data is None:
            self._aspect_data = self.load_file()
        return self._aspect_data

    @aspect_data.setter
    def aspect_data(self, value):
        self.dump_file(value)
        self._aspect_data = value

    async def act(self, action_name: str, msg: str | None) -> tuple[Action, str | None]:
        match action_name:
            case _:
                raise Exception(f'Unrecognized Action: {action_name}')

    async def handle_action(self, action: Action, message: str | None) -> MessageResult | QuitResult:
        if message and message.startswith('/'):
            input_parts = message.split(maxsplit=1)
            message = input_parts[1] if len(input_parts) > 1 else ''
            command = input_parts[0][1:]
            result = await action.handle_command(command, message)
        else:
            result = await action.handle(message)
        if isinstance(result, OperationResult):
            self.aspect_data, undo = apply_operation(self.aspect_data, result.op)
            return QuitResult(f'Finish Operation: {result.op}')
        return result

    def load_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def dump_file(self, data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)
