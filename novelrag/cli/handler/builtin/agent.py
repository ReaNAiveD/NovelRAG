import logging
from novelrag.agenturn.agent import RequestHandler
from novelrag.cli.handler.result import HandlerResult
from novelrag.cli.handler.handler import Handler
from novelrag.cli.command import Command


logger = logging.getLogger(__name__)


class AgentHandler(Handler):
    """Handler that delegates to the resource agent for handling user requests."""
    
    def __init__(self, agent: RequestHandler):
        self.agent = agent

    async def handle(self, command: Command) -> HandlerResult:
        message = command.message
        if message is None:
            raise ValueError("Message cannot be None for AgentHandler")

        response = await self.agent.handle_request(message)
        
        return HandlerResult(
            message=[response] if response else None,
        )
