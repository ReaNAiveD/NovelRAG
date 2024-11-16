import yaml

from novelrag.action import Action, QuitResult, MessageResult, OperationResult
from novelrag.operation import apply_operation


class AspectRegistry:
    """Registry to manage aspect-specific actions"""
    def __init__(self):
        self.actions = {}
    
    def register(self, name: str):
        """Decorator to register actions for a specific aspect"""
        def decorator(action_cls):
            self.actions[name] = action_cls
            action_cls._action_name = name
            return action_cls
        return decorator

class AspectContext:
    registry = None  # Override this in subclasses
    
    def __init__(self, name: str, file_path: str | None):
        if self.registry is None:
            raise NotImplementedError("Aspect must define a registry")
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

    async def act(self, action_name: str | None, msg: str | None) -> tuple[Action, str | None]:
        if action_name is None:
            # Handle default action case
            return await self.registry.actions['default'].create(msg, **self.action_config)
            
        if action_name in self.registry.actions:
            return await self.registry.actions[action_name].create(msg, **self.action_config)
            
        raise Exception(f'Unrecognized Action: {action_name}')

    async def handle_action(self, action: Action, command: str | None, message: str | None) -> MessageResult | QuitResult:
        result = await action.handle_command(command, message) if command else await action.handle(message)
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
