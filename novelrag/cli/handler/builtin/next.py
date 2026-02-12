import logging

from novelrag.agenturn.agent import AutonomousAgent
from novelrag.cli.command import Command
from novelrag.cli.handler.handler import Handler
from novelrag.cli.handler.result import HandlerResult
from novelrag.llm.logger import get_logger
from novelrag.llm import initialize_llm_logger

logger = logging.getLogger(__name__)
initialize_llm_logger(log_directory="logs")


class NextHandler(Handler):
    """Handler that triggers the autonomous agent to decide and pursue the next goal."""

    def __init__(self, autonomous_agent: AutonomousAgent):
        self.autonomous_agent = autonomous_agent

    async def handle(self, command: Command) -> HandlerResult:
        llm_logger = get_logger()
        if llm_logger:
            llm_logger.start_pursuit("Autonomous: /next")

        outcome = await self.autonomous_agent.pursue_next_goal()

        if llm_logger:
            llm_logger.complete_pursuit()
            try:
                log_file = llm_logger.dump_to_file()
                logger.debug(f"LLM logs saved to: {log_file}")
            except Exception as log_error:
                logger.warning(f"Failed to save LLM logs: {log_error}")

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
