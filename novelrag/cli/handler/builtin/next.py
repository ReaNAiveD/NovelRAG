import logging

from novelrag.agenturn.agent import AutonomousAgent
from novelrag.cli.command import Command
from novelrag.cli.handler.handler import Handler
from novelrag.cli.handler.result import HandlerResult

logger = logging.getLogger(__name__)


class NextHandler(Handler):
    """Handler that triggers the autonomous agent to decide and pursue the next goal."""

    def __init__(self, autonomous_agent: AutonomousAgent):
        self.autonomous_agent = autonomous_agent

    async def handle(self, command: Command) -> HandlerResult:
        outcome = await self.autonomous_agent.pursue_next_goal()

        if outcome is None:
            return HandlerResult(
                message=["No goals to pursue at this time."],
            )

        messages = []
        messages.append(f"Goal: {outcome.goal.description}")
        messages.append(f"Source: {outcome.goal.source}")
        messages.append(f"Status: {outcome.status.value}")
        if outcome.response:
            messages.append(f"Result: {outcome.response}")

        return HandlerResult(message=messages)
