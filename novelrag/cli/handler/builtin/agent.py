import logging

from novelrag.agenturn.agent import RequestHandler
from novelrag.cli.command import Command
from novelrag.cli.handler.interaction import InteractionHistory
from novelrag.cli.handler.handler import Handler
from novelrag.cli.handler.result import HandlerResult


logger = logging.getLogger(__name__)


class AgentHandler(Handler):
    """Handler that delegates to the resource agent for handling user requests."""
    
    def __init__(self, agent: RequestHandler, history: InteractionHistory | None = None):
        self.agent = agent
        self.history = history

    async def handle(self, command: Command) -> HandlerResult:
        message = command.message
        if message is None:
            raise ValueError("Message cannot be None for AgentHandler")

        outcome = await self.agent.handle_request(
            message,
            interaction_history=self.history,
        )

        return HandlerResult(
            message=[outcome.response] if outcome.response else None,
            details=outcome,
        )
