import json

from novelrag.intent import IntentContext, Action, LLMIntent, UpdateAction
from novelrag.exceptions import InvalidIndexError, InvalidMessageFormatError
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import PropertyOperation

class Update(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return 'update'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        try:
            split_msg = message.split(maxsplit=1)
            idx = int(split_msg[0])
            cnes = context.current_aspect.root_elements
            if idx >= len(cnes) or idx < 0:
                raise InvalidIndexError(idx, len(cnes), context.current_aspect.name)
            item = context.current_aspect.root_elements[idx]
            message = split_msg[1] if len(split_msg) > 1 else None
        except ValueError:
            raise InvalidMessageFormatError(
                self.name,
                context.aspect_name,
                message,
                "update INDEX [message]"
            )

        history = await context.conversation_history.get_messages(aspect='cne', intent='create')
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(f"[Update Premise #{idx}]{message}" if message else f"Update Premise #{idx}")

        template = context.template_env.load_template('cne/update.jinja2', lang=self.default_lang)
        system_prompt = template.render(
            session_summary=summary,
            related_info=related,
            target_cne=item,
            user_update_input=message,
            CNEs=context.current_aspect.root_elements,
        )

        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Update the specified CNE according to the guidelines.'}
        ], response_format='json_object')
        return Action(
            message=[resp],
            update=UpdateAction(
                item=PendingUpdateItem(
                    ops=[
                        PropertyOperation.new(
                            element_uri=item.uri,
                            data=json.loads(resp)
                        )
                    ]
                )
            )
        )
