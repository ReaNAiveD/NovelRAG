import logging
from dataclasses import dataclass

from novelrag.agenturn.channel import AgentChannel
from novelrag.agenturn.goal import LLMGoalTranslator
from novelrag.agenturn.types import AgentMessageLevel
from novelrag.cli.handler.builtin.next import NextHandler
from novelrag.cli.handler.builtin.quit import QuitHandler
from novelrag.cli.handler.builtin.redo import RedoHandler
from novelrag.cli.handler.builtin.undo import UndoHandler
from novelrag.config.novel_rag import NovelRagConfig
from novelrag.resource.repository import LanceDBResourceRepository
from novelrag.resource_agent import create_executor
from novelrag.resource_agent.backlog.local import LocalBacklog
from novelrag.resource_agent.goal_decider import CompositeGoalDecider
from novelrag.resource_agent.undo import LocalUndoQueue, MemoryUndoQueue, UndoQueue
from novelrag.cli.handler.builtin.agent import AgentHandler
from novelrag.cli.conversation import ConversationHistory
from novelrag.exceptions import HandlerNotFoundError, SessionQuitError
from novelrag.llm.factory import ChatLLMFactory, EmbeddingLLMFactory
from novelrag.cli.handler.registry import HandlerRegistry
from .command import Command

logger = logging.getLogger(__name__)


@dataclass
class Response:
    messages: list[str]
    redirect: Command | None = None


class SessionChannel(AgentChannel):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger

    async def send_message(self, content: str, level: AgentMessageLevel = AgentMessageLevel.INFO) -> None:
        match level:
            case AgentMessageLevel.DEBUG:
                self.logger.debug(content)
            case AgentMessageLevel.INFO:
                self.logger.info(content)
            case AgentMessageLevel.WARNING:
                self.logger.warning(content)
            case AgentMessageLevel.ERROR:
                self.logger.error(content)

    async def confirm(self, prompt: str) -> bool:
        result = input(prompt + 'y/N')
        if result.lower() in ['y', 'yes']:
            return True
        return False
    
    async def request(self, prompt: str) -> str:
        return input(prompt)


class Session:
    def __init__(
            self,
            *,
            handlers: HandlerRegistry,
            undo_queue: UndoQueue,
            conversation: ConversationHistory | None = None,
            chat_llm_factory: ChatLLMFactory | None = None,
            embedding_factory: EmbeddingLLMFactory | None = None,
    ):

        self.chat_llm_factory: ChatLLMFactory = chat_llm_factory or ChatLLMFactory()
        self.embedding_factory: EmbeddingLLMFactory = embedding_factory or EmbeddingLLMFactory()

        self.handler_registry: HandlerRegistry = handlers
        self.conversation: ConversationHistory = conversation or ConversationHistory.empty(chat_llm=self.chat_llm_factory.get())
        self.undo = undo_queue

    @classmethod
    async def from_config(cls, config: NovelRagConfig) -> 'Session':
        """Create a new session instance with configured components"""
        chat_llm = ChatLLMFactory.build(config.chat_llm)
        embedder = EmbeddingLLMFactory.build(config.embedding)
        chat_llm_factory = ChatLLMFactory(chat_llm)
        embedding_factory = EmbeddingLLMFactory(embedder)

        # Initialize repository
        repository = await LanceDBResourceRepository.from_config(
            config.resource_config,
            config.vector_store,
            embedder,
            config.default_resource_dir,
        )
        channel = SessionChannel(logger)
        undo_queue = LocalUndoQueue(config.undo_path) if config.undo_path else MemoryUndoQueue()
        conversation_history = ConversationHistory.empty(chat_llm=chat_llm)
        backlog = LocalBacklog(config.backlog_path) if config.backlog_path else None

        agent = create_executor(
            resource_repo=repository,
            channel=channel,
            chat_llm=chat_llm,
            beliefs=config.agent_beliefs,
            lang=config.template_lang,
            backlog=backlog,
            undo_queue=undo_queue,
        )
        goal_translator = LLMGoalTranslator(chat_llm, lang=config.template_lang or "en")
        agent_request_handler = agent.create_request_handler(goal_translator)
        agent_handler = AgentHandler(agent_request_handler)

        # Create autonomous agent with CompositeGoalDecider
        goal_decider = CompositeGoalDecider(
            repo=repository,
            chat_llm=chat_llm,
            template_lang=config.template_lang or "en",
            backlog=backlog,
            undo_queue=undo_queue,
        )
        autonomous_agent = agent.create_autonomous_agent(goal_decider)
        next_handler = NextHandler(autonomous_agent)

        undo_handler = UndoHandler(resource_repo=repository, undo_queue=undo_queue)
        redo_handler = RedoHandler(resource_repo=repository, undo_queue=undo_queue)
        quit_handler = QuitHandler()
        handlers = HandlerRegistry(
            _default=agent_handler,
            next=next_handler,
            undo=undo_handler,
            redo=redo_handler,
            quit=quit_handler,
        )

        # Create session with configured handlers
        return cls(
            handlers=handlers,
            undo_queue=undo_queue,
            conversation=conversation_history,
            chat_llm_factory=chat_llm_factory,
            embedding_factory=embedding_factory,
        )

    async def invoke(self, command: Command):
        messages = []
        handler = None
        if self.handler_registry:
            handler = await self.handler_registry.get(command.handler or '_default')
        if not handler:
            raise HandlerNotFoundError(command.handler or '_default')

        result = await handler.handle(command)

        self.conversation.add_user(
            command.text,
            intent=command.handler,
        )

        if result.message:
            self.conversation.add_assistant(
                '\n'.join(result.message),
                intent=command.handler,
            )
            messages.extend(result.message)
        if result.quit:
            raise SessionQuitError()
        if result.redirect:
            redirect = Command(
                redirect_source=command.raw or command.text,
                handler=command.handler,
                message=command.message,
                is_redirect=True,
            )
        else:
            redirect = None
        return Response(messages=messages, redirect=redirect)
