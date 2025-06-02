import json
import logging

from novelrag.intent import IntentContext, Action, LLMIntent, UpdateAction
from novelrag.intent.action import Redirect
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import validate_op

logger = logging.getLogger(__name__)


def _get_code_block(msg: str):
    parts = msg.split("```", maxsplit=2)
    if len(parts) == 3:
        return parts[1].split('\n', maxsplit=1)[-1].rsplit('\n', maxsplit=1)[0]
    return None


def _patch_location(op: dict):
    if 'target' not in op:
        logger.warning(f"Missing `target` field in Operation.")
        return None
    if op['target'] == 'element':
        op['location'] = {
            'type': 'aspect',
            'aspect': 'cne',
        }
    return op


class Default(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return '_default'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        if not message:
            return Action(
                redirect=Redirect(
                    aspect='cne',
                    intent='list',
                    message=None,
                )
            )

        history = await context.conversation_history.get_messages(aspect='cne', intent=None)
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(message)

        template = context.template_env.load_template('cne/default.jinja2', lang=self.default_lang)
        system_prompt = template.render(
            user_input=message,
            session_summary=summary,
            related_info=related,
            CNEs=context.current_aspect.root_elements,
        )

        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Create a new CNE according to the guidelines.'}
        ])
        code_block = _get_code_block(resp)

        r_message = [resp]
        ops = None
        try:
            if code_block:
                update = json.loads(code_block)
                if isinstance(update, list):
                    ops = [_patch_location(op) for op in update]
                    ops = [validate_op(_patch_location(op)) for op in ops if ops]
                elif isinstance(update, dict):
                    op = _patch_location(update)
                    if op:
                        ops = [validate_op(op)]
                else:
                    r_message.append("Update Ignored: Code Block is not a valid Operation.")
        except json.JSONDecodeError as e:
            logger.warning("Failed to decode generated operations", exc_info=e)
            r_message.append("Update Ignored: Code Block is not a valid JSON.")
        if ops:
            return Action(
                message=[resp],
                submit=False,
                update=UpdateAction(
                    overwrite_pending=True,
                    item=PendingUpdateItem(
                        ops=ops
                    )
                )
            )
        else:
            return Action(
                message=r_message,
            )
