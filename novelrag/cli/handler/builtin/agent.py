import logging
from novelrag.agenturn.agent import RequestHandler
from novelrag.llm.logger import get_logger
from novelrag.llm import initialize_llm_logger
from novelrag.cli.handler.result import HandlerResult
from novelrag.cli.handler.handler import Handler
from novelrag.cli.command import Command


logger = logging.getLogger(__name__)
# Call this during application startup
initialize_llm_logger(log_directory="logs")


class AgentHandler(Handler):
    """Handler that delegates to the resource agent for handling user requests."""
    
    def __init__(self, agent: RequestHandler):
        self.agent = agent

    async def handle(self, command: Command) -> HandlerResult:
        message = command.message
        if message is None:
            raise ValueError("Message cannot be None for AgentHandler")

        llm_logger = get_logger()
        if llm_logger:
            llm_logger.start_pursuit(f"Request: {message}")

        response = await self.agent.handle_request(message)

        if llm_logger:
            llm_logger.complete_pursuit()
            try:
                log_file = llm_logger.dump_to_file()
                logger.debug(f"LLM logs saved to: {log_file}")
            except Exception as log_error:
                logger.warning(f"Failed to save LLM logs: {log_error}")
        
        return HandlerResult(
            message=[response] if response else None,
        )
