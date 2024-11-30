from novelrag.core.action import Action, QuitResult, MessageResult, OperationResult
from novelrag.core.storage import AspectStorage

class AspectContext:
    registry = None  # Override this in subclasses
    
    def __init__(self, name: str, data: AspectStorage):
        if self.registry is None:
            raise NotImplementedError("Aspect must define a registry")
        self.name = name
        self.data = data

    @property
    def prompt_section(self):
        return self.name

    @property
    def action_config(self):
        return {}

    async def act(self, action_name: str | None, msg: str | None) -> tuple[Action, str | None]:
        if action_name is None:
            # Handle default action case
            return await self.registry.actions['_default'].create(msg, **self.action_config)
            
        if action_name in self.registry.actions:
            return await self.registry.actions[action_name].create(msg, **self.action_config)
            
        raise Exception(f'Unrecognized Action: {action_name}')

    async def handle_action(self, action: Action, command: str | None, message: str | None) -> MessageResult | QuitResult:
        result = await action.handle_command(command, message) if command else await action.handle(message)
        if isinstance(result, OperationResult):
            undo = self.data.apply(result.op)
            return QuitResult(f'Finish Operation: {result.op}')
        return result
