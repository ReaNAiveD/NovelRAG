import json
import logging

from novelrag.intent import IntentContext, Action, UpdateAction, Intent
from novelrag.exceptions import InvalidIndexError, InvalidMessageFormatError
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import ElementOperation, AspectLocation

logger = logging.getLogger(__name__)


class Delete(Intent):
    @property
    def default_name(self) -> str | None:
        return 'delete'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        try:
            idx = int(message)
            cnes = context.current_aspect.root_elements
            if idx >= len(cnes) or idx < 0:
                raise InvalidIndexError(idx, len(cnes), context.current_aspect.name)
            item = context.current_aspect.root_elements[idx]
            message = f"Delete (Submit to Confirm): {json.dumps(item.children_dict(), ensure_ascii=False)}"
        except ValueError as e:
            raise InvalidMessageFormatError(self.name, context.current_aspect.name, 'delete [INDEX]') from e
        return Action(
            message=[message],
            update=UpdateAction(
                item=PendingUpdateItem(
                    ops=ElementOperation.new(
                        location=AspectLocation.new('cne'),
                        start=idx,
                        end=idx+1,
                    )
                )
            )
        )
