from novelrag.cli.handler.result import HandlerResult
from novelrag.cli.handler.handler import Handler
from novelrag.cli.command import Command
from novelrag.resource.operation import validate_op
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue


class RedoHandler(Handler):
    def __init__(self, resource_repo: ResourceRepository, undo_queue: UndoQueue) -> None:
        self.resource_repo = resource_repo
        self.undo_queue = undo_queue

    async def handle(self, command: Command) -> HandlerResult:
        redo_tasks = self.undo_queue.pop_redo_group()
        if not redo_tasks:
            return HandlerResult(
                message=["No actions to redo."],
            )
        for task in redo_tasks:
            match task.method:
                case 'apply':
                    op = task.params.get('op')
                    op = validate_op(op) # type: ignore
                    redo = await self.resource_repo.apply(op)
                    self.undo_queue.add_undo_item(ReversibleAction(method='apply', params={'op': redo.model_dump()}, group=task.group), clear_redo=False)
                case 'update_relationships':
                    source_uri = task.params['source_uri']
                    target_uri = task.params['target_uri']
                    relationships = task.params.get('relationships', [])
                    undo_relationships = await self.resource_repo.update_relationships(source_uri, target_uri, relationships)
                    self.undo_queue.add_undo_item(ReversibleAction(method='update_relationships', params={
                        'source_uri': source_uri,
                        'target_uri': target_uri,
                        'relationships': undo_relationships}, group=task.group), clear_redo=False)
                case 'remove_aspect':
                    name = task.params['name']
                    aspect = self.resource_repo.remove_aspect(name)
                    if aspect:
                        self.undo_queue.add_undo_item(ReversibleAction(method='add_aspect', params={'name': name, 'metadata': aspect.to_config().model_dump()}, group=task.group), clear_redo=False)
                    else:
                        return HandlerResult(
                            message=[f"Aspect '{name}' does not exist, redo operation skipped."],
                        )
                case 'add_aspect':
                    name = task.params['name']
                    metadata = task.params['metadata']
                    self.resource_repo.add_aspect(name, metadata)
                    self.undo_queue.add_undo_item(ReversibleAction(method='remove_aspect', params={'name': name}, group=task.group), clear_redo=False)
                case _:
                    return HandlerResult(
                        message=[f"Unknown redo method '{task.method}', skipping."],
                    )
        if len(redo_tasks) == 1:
            return HandlerResult(
                message=[f"Redid action: {redo_tasks[0].method}"],
            )
        else:
            return HandlerResult(
                message=[f"Redid {len(redo_tasks)} actions."],
            )
