import json

from novelrag.intent import LLMIntent, IntentContext
from novelrag.intent.action import Action, UpdateAction
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import ResourceOperation, ResourceLocation


class Create(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return 'create'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        cnes = context.current_aspect.root_elements

        history = await context.conversation_history.get_messages(aspect='cne', intent='create')
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(f"[Create Premise]{message}" if message else "Create Premise")

        template = context.template_env.load_template('cne/create.jinja2', lang=self.default_lang)
        system_prompt = template.render(
            session_summary=summary,
            related_info=related,
            user_input=message,
            CNEs=cnes,
        )

        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Create a new CNE according to the guidelines.'}
        ], response_format='json_object')
        return Action(
            message=[resp],
            update=UpdateAction(
                item=PendingUpdateItem(
                    ops=[
                        ResourceOperation.new(
                            location=ResourceLocation.aspect('cne'),
                            start=len(cnes),
                            data=[json.loads(resp)]
                        )
                    ]
                )
            )
        )
