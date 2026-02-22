from novelrag.cli.command import Command
from novelrag.cli.handler.interaction import UndoRedoDetails
from novelrag.cli.handler.handler import Handler
from novelrag.cli.handler.result import HandlerResult
from novelrag.resource.operation import validate_op
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue


class UndoHandler(Handler):
    def __init__(self, resource_repo: ResourceRepository, undo_queue: UndoQueue) -> None:
        self.resource_repo = resource_repo
        self.undo_queue = undo_queue

    async def handle(self, command: Command) -> HandlerResult:
        undo_tasks = self.undo_queue.pop_undo_group()
        if not undo_tasks:
            return HandlerResult(
                message=["No actions to undo."],
            )
        for task in undo_tasks:
            match task.method:
                case 'apply':
                    op = task.params.get('op')
                    op = validate_op(op) # type: ignore
                    redo = await self.resource_repo.apply(op)
                    self.undo_queue.add_redo_item(ReversibleAction(method='apply', params={'op': redo.model_dump()}, group=task.group))
                case 'update_relationships':
                    source_uri = task.params['source_uri']
                    target_uri = task.params['target_uri']
                    relationships = task.params.get('relationships', [])
                    redo_relationships = await self.resource_repo.update_relationships(source_uri, target_uri, relationships)
                    self.undo_queue.add_redo_item(ReversibleAction(method='update_relationships', params={
                        'source_uri': source_uri,
                        'target_uri': target_uri,
                        'relationships': redo_relationships}, group=task.group))
                case 'remove_aspect':
                    name = task.params['name']
                    aspect = self.resource_repo.remove_aspect(name)
                    if aspect:
                        self.undo_queue.add_redo_item(ReversibleAction(method='add_aspect', params={'name': name, 'metadata': aspect.to_config().model_dump()}, group=task.group))
                    else:
                        return HandlerResult(
                            message=[f"Aspect '{name}' does not exist, undo operation skipped."],
                        )
                case 'add_aspect':
                    name = task.params['name']
                    metadata = task.params['metadata']
                    self.resource_repo.add_aspect(name, metadata)
                    self.undo_queue.add_redo_item(ReversibleAction(method='remove_aspect', params={'name': name}, group=task.group))
                case _:
                    return HandlerResult(
                        message=[f"Unknown undo method '{task.method}', skipping."],
                    )
        details = UndoRedoDetails(
            action="undo",
            methods=[t.method for t in undo_tasks],
            count=len(undo_tasks),
            descriptions=[t.description for t in undo_tasks],
        )
        if len(undo_tasks) == 1:
            return HandlerResult(
                message=[f"Undid action: {undo_tasks[0].method}"],
                details=details,
            )
        else:
            return HandlerResult(
                message=[f"Undid {len(undo_tasks)} actions."],
                details=details,
            )
