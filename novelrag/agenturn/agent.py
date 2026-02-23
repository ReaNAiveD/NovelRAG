"""Agent implementation using OrchestrationLoop for strategic execution."""

import logging
from datetime import datetime
from typing import Any

from novelrag.agenturn.goal import Goal, GoalDecider, GoalTranslator
from novelrag.agenturn.procedure import ExecutionContext
from novelrag.agenturn.pursuit import ActionDeterminer, PursuitOutcome, PursuitProgress, PursuitStatus
from novelrag.agenturn.step import OperationPlan, OperationOutcome, Resolution, StepStatus
from novelrag.agenturn.tool import SchematicTool, ToolResult, ToolError
from novelrag.agenturn.interaction import InteractionContext
from novelrag.tracer import trace_intent, trace_pursuit, trace_tool

logger = logging.getLogger(__name__)


class GoalExecutor:
    """Core agent that executes a single goal. No translation, no looping."""

    def __init__(
        self,
        beliefs: list[str],
        tools: dict[str, SchematicTool],
        determiner: ActionDeterminer,
        channel: ExecutionContext,
    ):
        self.beliefs = beliefs
        self.tools = tools
        self.determiner = determiner
        self.channel = channel

    @trace_pursuit("handle_goal")
    async def handle_goal(
            self,
            goal: Goal,
            interaction_history: InteractionContext | None = None,
    ) -> PursuitOutcome:
        await self.channel.info(f"Starting handle goal: {goal}")

        pursuit_progress = PursuitProgress(goal=goal)

        try:
            while True:
                directive = await self.determiner.determine_action(
                    beliefs=self.beliefs,
                    pursuit_progress=pursuit_progress,
                    available_tools=self.tools,
                    ctx=self.channel,
                    interaction_history=interaction_history,
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

    @trace_tool()
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

        try:
            await self.channel.debug(f"Calling tool {tool_name} with params: {params}")
            result = await tool.call(self.channel, **params)

            if isinstance(result, ToolResult):
                return OperationOutcome(
                    operation=step,
                    status=StepStatus.SUCCESS,
                    result=result.result,
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
    
    @trace_intent("handle_request")
    async def handle_request(
            self,
            request: str,
            interaction_history: InteractionContext | None = None,
    ) -> PursuitOutcome:
        """Translate request to goal, execute, and return response."""        
        goal = await self.goal_translator.translate(
            request, self.executor.beliefs,
            interaction_history=interaction_history,
        )
        outcome = await self.executor.handle_goal(
            goal, interaction_history=interaction_history,
        )
        return outcome


class AutonomousAgent:
    """Agent that autonomously generates and pursues goals."""

    def __init__(
        self,
        executor: GoalExecutor,
        goal_decider: GoalDecider,
    ):
        self.executor = executor
        self.goal_decider = goal_decider
    
    @trace_intent("autonomous_pursuit")
    async def pursue_next_goal(
            self,
            interaction_history: InteractionContext | None = None,
    ) -> PursuitOutcome | None:
        """Decide on the next goal and pursue it."""
        goal = await self.goal_decider.next_goal(
            self.executor.beliefs,
            interaction_history=interaction_history,
        )
        if goal is None:
            await self.executor.channel.info("No new goals to pursue.")
            return None
        await self.executor.channel.info(f"Decided to pursue new goal: {goal.description}")
        outcome = await self.executor.handle_goal(
            goal, interaction_history=interaction_history,
        )
        return outcome
