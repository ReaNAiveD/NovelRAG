import logging
from novelrag.agenturn.agent import RequestHandler
from novelrag.agenturn.goal import LLMGoalTranslator
from novelrag.llm.logger import get_logger
from novelrag.resource.repository import ResourceRepository
from novelrag.agenturn.channel import SessionChannel
from novelrag.resource_agent import create_executor
from novelrag.intent import LLMIntent, IntentContext, Action
from novelrag.llm import initialize_logger


logger = logging.getLogger(__name__)
# Call this during application startup
initialize_logger(log_directory="logs")


class AgentIntent(LLMIntent):
    """Intent that delegates to the resource agent for handling user requests."""
    
    def __init__(self, *, resource_repo: ResourceRepository, name: str | None, chat_llm: dict | None = None, lang: str | None = None, channel: SessionChannel, **kwargs):
        super().__init__(name=name, chat_llm=chat_llm, lang=lang, **kwargs)
        self.resource_repo = resource_repo
        self.channel = channel
        self.lang = lang or "en"

    @property
    def default_name(self) -> str | None:
        return 'agent'
    
    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        if message is None:
            raise ValueError("Message cannot be None for AgentIntent")

        # Create agent using the factory function from resource_agent
        executor = create_executor(
            resource_repo=self.resource_repo,
            channel=self.channel,
            chat_llm=self.chat_llm(context.chat_llm_factory),
            beliefs=[],
            lang=self.lang,
        )
        goal_translator = LLMGoalTranslator(chat_llm=self.chat_llm(context.chat_llm_factory), lang=self.lang)
        request_handler = RequestHandler(executor, goal_translator)

        llm_logger = get_logger()
        if llm_logger:
            llm_logger.start_pursuit(f"Request: {message}")

        response = await request_handler.handle_request(message)

        if llm_logger:
            llm_logger.complete_pursuit()
            try:
                log_file = llm_logger.dump_to_file()
                await self.channel.debug(f"LLM logs saved to: {log_file}")
            except Exception as log_error:
                logger.warning(f"Failed to save LLM logs: {log_error}")
        
        # Get any accumulated output from the channel
        channel_output = self.channel.get_output()
        
        # Combine agent result with channel output, ensuring list format
        messages = []
        if channel_output:
            if isinstance(channel_output, list):
                messages.extend(channel_output)
            else:
                messages.append(channel_output)
        if response:
            messages.append(response)
        
        return Action(
            message=messages if messages else None,
        )
