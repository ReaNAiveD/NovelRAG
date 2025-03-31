import logging
from dataclasses import dataclass

from novelrag.conversation import ConversationHistory
from novelrag.exceptions import IntentNotFoundError, SessionQuitError, NoItemToSubmitError, NoItemToUndoError
from novelrag.intent import IntentFactory, DictionaryIntentFactory, IntentContext
from novelrag.llm.factory import ChatLLMFactory, EmbeddingLLMFactory
from novelrag.pending_queue import PendingUpdateQueue
from novelrag.resource import ResourceRepository
from .command import Command
from .context import Context, AspectFactory
from .undo import UndoQueue


logger = logging.getLogger(__name__)


@dataclass
class Response:
    messages: list[str]
    redirect: Command | None = None


class Session:
    def __init__(
            self,
            *,
            aspect_factory: AspectFactory,
            conversation: ConversationHistory | None = None,
            update_queue: PendingUpdateQueue | None = None,
            resource_repository: ResourceRepository | None = None,
            intents: IntentFactory | None = None,
            chat_llm_factory: ChatLLMFactory | None = None,
            embedding_factory: EmbeddingLLMFactory | None = None,
    ):
        self.chat_llm_factory: ChatLLMFactory = chat_llm_factory or ChatLLMFactory()
        self.embedding_factory: EmbeddingLLMFactory = embedding_factory or EmbeddingLLMFactory()

        self.context = Context(aspect_factory=aspect_factory)
        self.intent_registry: IntentFactory = intents or DictionaryIntentFactory([])
        self.conversation: ConversationHistory = conversation or ConversationHistory.empty(chat_llm=self.chat_llm_factory.get())
        self.update_queue: PendingUpdateQueue = update_queue or PendingUpdateQueue()
        self.resource_repository = resource_repository
        self.undo = UndoQueue()

    async def invoke(self, command: Command):
        messages = []
        if command.aspect:
            await self.context.switch(aspect=command.aspect)
            logger.info(f'Switched to new Aspect: {command.aspect}')

        if not command.intent:
            return Response(messages=messages)

        intent = None
        if self.context.cur_aspect:
            intent = await self.context.cur_aspect.intents.get_intent(command.intent)
        if not intent and self.intent_registry:
            intent = await self.intent_registry.get_intent(command.intent)
        if not intent:
            raise IntentNotFoundError(command.intent or '_default')

        output = await intent.handle(
            command.message,
            context=IntentContext(
                current_element=self.context.cur_element,
                current_aspect=self.context.cur_aspect.data if self.context.cur_aspect else None,
                raw_command=command.raw,
                pending_update=self.update_queue,
                resource_repository=self.resource_repository,
                chat_llm_factory=self.chat_llm_factory,
                embedding_factory=self.embedding_factory,
                conversation_history=self.conversation,
            )
        )

        output.verify()

        self.conversation.add_user(
            command.text,
            aspect=self.context.cur_aspect.name if self.context.cur_aspect else None,
            intent=intent.name if intent else None,
        )

        if output.cd:
            await self.context.cd(output.cd)
            logger.info(f"Change to new Path '{output.cd}'")
        if output.message:
            self.conversation.add_assistant(
                '\n'.join(output.message),
                aspect=self.context.cur_aspect.name if self.context.cur_aspect else None,
                intent=intent.name if intent else None,
            )
            messages.extend(output.message)
        if output.update:
            if output.update.overwrite_pending:
                self.update_queue.clear()
                logger.info("Clear Pending Update Queue")
            self.update_queue.push(output.update.item)
            logger.info(f"New Pending Update with {len(output.update.item.ops)} Operations")
        if output.submit:
            update = self.update_queue.lpop()
            if update:
                undo = []
                for op in update.ops:
                    undo.append(await self.resource_repository.apply(op))
                undo = undo[::-1]
                self.undo.push(undo, update)
                logger.info(f"Submit Update with {len(update.ops)} Operations")
            else:
                raise NoItemToSubmitError()
        if output.undo:
            undo = self.undo.pop()
            if undo:
                # Always clear Update Queue when undo
                self.update_queue.clear()
                for op in undo.ops:
                    _ = await self.resource_repository.apply(op)
                self.update_queue.lpush(undo.redo)
                logger.info(f"Undo with {len(undo.ops)} Operations")
            else:
                raise NoItemToUndoError()
        if output.quit:
            if self.context.cur_aspect:
                await self.context.switch(None)
                logger.info(f'Quit current Aspect')
            else:
                raise SessionQuitError()
        if output.redirect:
            redirect = Command(
                redirect_from=command.raw or command.text,
                aspect=command.aspect,
                intent=command.intent,
                message=command.message,
            )
        else:
            redirect = None
        return Response(messages=messages, redirect=redirect)
