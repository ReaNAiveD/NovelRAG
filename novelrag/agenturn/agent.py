"""Agent implementation using OrchestrationLoop for strategic execution."""

import logging
from datetime import datetime
from typing import Any

from novelrag.agenturn.channel import AgentChannel
from novelrag.agenturn.goal import Goal, GoalDecider, GoalTranslator
from novelrag.agenturn.pursuit import ActionDeterminer, PursuitAssessment, PursuitAssessor, PursuitOutcome, PursuitProgress, PursuitStatus
from novelrag.agenturn.step import OperationPlan, OperationOutcome, Resolution, StepStatus
from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.llm import get_logger

logger = logging.getLogger(__name__)


class AgentToolRuntime(ToolRuntime):
    """Agent's implementation of ToolRuntime that routes calls to AgentChannel."""
    
    def __init__(self, channel: AgentChannel):
        self.channel = channel
        self._backlog: list[str] = []
        self._progress: dict[str, list[str]] = {}
        self._triggered_actions: list[dict[str, str]] = []

    async def debug(self, content: str):
        await self.channel.debug(content)

    async def message(self, content: str):
        await self.channel.info(content)

    async def warning(self, content: str):
        await self.channel.warning(content)

    async def error(self, content: str):
        await self.channel.error(content)

    async def confirmation(self, prompt: str) -> bool:
        return await self.channel.confirm(prompt)

    async def user_input(self, prompt: str) -> str:
        return await self.channel.request(prompt)

    async def progress(self, key: str, value: str, description: str | None = None):
        if key not in self._progress:
            self._progress[key] = []
        self._progress[key].append(value)

    async def backlog(self, content: dict, priority: str | None = None):
        self._backlog.append(str(content))

    def take_backlog(self) -> list[str]:
        """Take all backlog items, clearing the internal list."""
        backlog = self._backlog
        self._backlog = []
        return backlog

    def take_triggered_actions(self) -> list[dict[str, str]]:
        """Take all triggered actions, clearing the internal list."""
        actions = self._triggered_actions
        self._triggered_actions = []
        return actions


class GoalExecutor:
    """Core agent that executes a single goal. No translation, no looping."""

    def __init__(
        self,
        beliefs: list[str],
        tools: dict[str, SchematicTool],
        determiner: ActionDeterminer,
        channel: AgentChannel,
    ):
        self.beliefs = beliefs
        self.tools = tools
        self.determiner = determiner
        self.channel = channel

    async def handle_goal(self, goal: Goal) -> PursuitOutcome:
        await self.channel.info(f"Starting handle goal: {goal}")

        pursuit_progress = PursuitProgress(goal=goal)

        try:
            while True:
                directive = await self.determiner.determine_action(
                    beliefs=self.beliefs,
                    pursuit_progress=pursuit_progress,
                    available_tools=self.tools
                )
                if isinstance(directive, Resolution):
                    # Goal pursuit is complete
                    await self.channel.info(f"Goal pursuit resolved: {directive.reason}")
                    return PursuitOutcome(
                        goal=goal,
                        reason=directive.reason,
                        response=directive.response,
                        status=PursuitStatus.COMPLETED,
                        executed_steps=pursuit_progress.executed_steps,
                        resolution=directive,
                        resolve_at=datetime.now()
                    )
                elif isinstance(directive, OperationPlan):
                    # Execute the recommended tool
                    await self.channel.info(f"Executing: {directive.reason}")
                else:
                    raise RuntimeError("Unknown directive type")
                outcome = await self._execute_tool(
                    tool_name=directive.tool,
                    params=directive.parameters,
                    reason=directive.reason,
                )
                pursuit_progress.executed_steps.append(outcome)
                # Log execution result
                if outcome.status == StepStatus.SUCCESS:
                    await self.channel.info(f"✓ Completed: {outcome.operation.reason}")
                else:
                    await self.channel.error(f"✗ Failed: {outcome.operation.reason} - {outcome.error_message}")
        except Exception as e:
            error_msg = f"Goal pursuit failed with error: {str(e)}"
            await self.channel.error(error_msg)
            logger.exception("Error during goal pursuit")
            return PursuitOutcome(
                goal=goal,
                reason="Pursuit failed",
                response=error_msg,
                status=PursuitStatus.FAILED,
                executed_steps=pursuit_progress.executed_steps,
                resolution=Resolution(
                    reason="Pursuit failed",
                    response=error_msg,
                    status="failed"
                ),
                resolve_at=datetime.now()
            )

    async def _execute_tool(self, tool_name: str, params: dict[str, Any], reason: str) -> OperationOutcome:
        """Execute a single tool and return the outcome."""
        start_time = datetime.now()
        
        # Create step definition for tracking
        step = OperationPlan(reason=reason, tool=tool_name, parameters=params)
        
        # Check if tool exists
        if tool_name not in self.tools:
            return OperationOutcome(
                operation=step,
                status=StepStatus.FAILED,
                error_message=f"Tool {tool_name} not found",
                started_at=start_time,
                completed_at=datetime.now()
            )

        tool = self.tools[tool_name]
        runtime = AgentToolRuntime(self.channel)

        try:
            await self.channel.debug(f"Calling tool {tool_name} with params: {params}")
            result = await tool.call(runtime, **params)
            
            from novelrag.agenturn.tool.types import ToolResult, ToolError
            
            if isinstance(result, ToolResult):
                return OperationOutcome(
                    operation=step,
                    status=StepStatus.SUCCESS,
                    results=[result.result],
                    started_at=start_time,
                    completed_at=datetime.now(),
                )
            elif isinstance(result, ToolError):
                return OperationOutcome(
                    operation=step,
                    status=StepStatus.FAILED,
                    error_message=result.error_message,
                    started_at=start_time,
                    completed_at=datetime.now(),
                )
            else:
                raise ValueError(f"Unexpected tool result type: {type(result)}")
                
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return OperationOutcome(
                operation=step,
                status=StepStatus.FAILED,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now(),
            )
    
    def create_request_handler(self, goal_translator: GoalTranslator) -> 'RequestHandler':
        """Create a RequestHandler that uses this executor."""
        return RequestHandler(executor=self, goal_translator=goal_translator)

    def create_autonomous_agent(self, goal_decider: GoalDecider) -> 'AutonomousAgent':
        """Create an AutonomousAgent that uses this executor."""
        return AutonomousAgent(executor=self, goal_decider=goal_decider)


class RequestHandler:
    """Handles incoming requests to the agent."""

    def __init__(self, executor: GoalExecutor, goal_translator: GoalTranslator):
        self.executor = executor
        self.goal_translator = goal_translator
    
    async def handle_request(self, request: str) -> str:
        """Translate request to goal, execute, and return response."""        
        goal = await self.goal_translator.translate(request, self.executor.beliefs)
        outcome = await self.executor.handle_goal(goal)
        return outcome.response


class AutonomousAgent:
    """Agent that autonomously generates and pursues goals."""

    def __init__(
        self,
        executor: GoalExecutor,
        goal_decider: GoalDecider,
    ):
        self.executor = executor
        self.goal_decider = goal_decider
    
    async def pursue_next_goal(self) -> PursuitOutcome | None:
        """Decide on the next goal and pursue it."""
        goal = await self.goal_decider.next_goal(self.executor.beliefs)
        if goal is None:
            await self.executor.channel.info("No new goals to pursue.")
            return None
        await self.executor.channel.info(f"Decided to pursue new goal: {goal.description}")
        outcome = await self.executor.handle_goal(goal)
        return outcome
