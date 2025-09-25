import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .channel import AgentChannel
from .context import PursuitContext
from .steps import StepDefinition, StepOutcome, StepStatus
from .tool import ContextualTool, ToolRuntime, LLMLogicalOperationTool
from .types import ToolResult, ToolDecomposition, ToolError

logger = logging.getLogger(__name__)


class ExecutionToolRuntime(ToolRuntime):
    def __init__(self, channel: AgentChannel | None = None):
        self.channel = channel
        self._backlog: list[str] = []
        self._progress: dict[str, list[str]] = {}
        self._triggered_actions: list[dict[str, str]] = []

    async def debug(self, content: str):
        if self.channel:
            await self.channel.debug(content)
        else:
            logger.debug(content)

    async def message(self, content: str):
        if self.channel:
            await self.channel.info(content)
        else:
            logger.info(content)

    async def warning(self, content: str):
        if self.channel:
            await self.channel.warning(content)
        else:
            logger.warning(content)

    async def error(self, content: str):
        if self.channel:
            await self.channel.error(content)
        else:
            logger.error(content)

    async def confirmation(self, prompt: str):
        if self.channel:
            return await self.channel.confirm(prompt)
        else:
            raise RuntimeError("Confirmation required but no channel provided")

    async def user_input(self, prompt: str):
        if self.channel:
            return await self.channel.request(prompt)
        else:
            raise RuntimeError("User input required but no channel provided")

    async def progress(self, key: str, value: str, description: str | None = None):
        if key not in self._progress:
            self._progress[key] = []
        self._progress[key].append(value)

    async def trigger_action(self, action: dict[str, str]):
        self._triggered_actions.append(action)

    async def backlog(self, content: dict, priority: str | None = None):
        self._backlog.append(json.dumps(content))


# Execute step function - works directly with StepDefinition
async def _execute_step(step: StepDefinition, tools: dict[str, ContextualTool], believes: list[str],
                        context: list[str], channel: AgentChannel | None = None,
                        fallback_tool: LLMLogicalOperationTool | None = None) -> StepOutcome:
    """Execute the action and return its outcome."""
    start_time = datetime.now()
    # Get the tool, use fallback if tool is None or not found
    tool = None
    if step.tool and step.tool in tools:
        tool = tools[step.tool]
    elif step.tool is None or step.tool == '' or step.tool not in tools:
        if fallback_tool:
            tool = fallback_tool
            if channel:
                await channel.debug(f"Using fallback tool({tool.name}) for step with tool='{step.tool}' - performing logical operation: {step.intent}")
        else:
            return StepOutcome(
                action=step,
                status=StepStatus.FAILED,
                started_at=start_time,
                error_message=f"Tool {step.tool} not found and no fallback tool available.",
                completed_at=datetime.now()
            )

    if not tool:
        return StepOutcome(
            action=step,
            status=StepStatus.FAILED,
            started_at=start_time,
            error_message=f"Tool {step.tool} not found.",
            completed_at=datetime.now()
        )

    try:
        # Initialize tool call
        runtime = ExecutionToolRuntime(channel=channel)
        result = await tool.call(
            runtime=runtime,
            believes=believes,
            step=step,
            context=context,
            tools={name: t.description or '' for name, t in tools.items()}
        )

        if isinstance(result, ToolDecomposition):
            return StepOutcome(
                action=step,
                status=StepStatus.DECOMPOSED,
                started_at=start_time,
                completed_at=datetime.now(),
                decomposed_actions=result.steps,
                rerun=result.rerun,
                triggered_actions=runtime._triggered_actions,
                backlog_items=runtime._backlog,
                progress=runtime._progress,
            )
        elif isinstance(result, ToolResult):
            return StepOutcome(
                action=step,
                status=StepStatus.SUCCESS,
                results=[result.result],
                started_at=start_time,
                completed_at=datetime.now(),
                triggered_actions=runtime._triggered_actions,
                backlog_items=runtime._backlog,
                progress=runtime._progress,
            )
        elif isinstance(result, ToolError):
            return StepOutcome(
                action=step,
                status=StepStatus.FAILED,
                error_message=result.error_message,
                started_at=start_time,
                completed_at=datetime.now(),
                triggered_actions=runtime._triggered_actions,
                backlog_items=runtime._backlog,
                progress=runtime._progress,
            )
        else:
            raise ValueError(f"Unexpected tool output type: {type(result)}")
    except Exception as e:
        logging.warning(f"Error executing step [{step.intent}] with tool [{step.tool}]: {e}", exc_info=True)
        return StepOutcome(
            action=step,
            status=StepStatus.FAILED,
            error_message=str(e),
            started_at=start_time,
            completed_at=datetime.now(),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Represents the agent's evolving plan to achieve a goal.

    Unlike a schedule which implies fixed timing, an execution plan
    adapts dynamically as new information emerges.
    """
    goal: str
    pending_steps: list[StepDefinition] = field(default_factory=list)
    executed_steps: list[StepOutcome] = field(default_factory=list)

    def finished(self) -> bool:
        """Check if there are actions waiting to be executed."""
        return len(self.pending_steps) == 0

    def __str__(self) -> str:
        """String representation of the execution plan."""
        lines = [f"ExecutionPlan: {self.goal}"]

        # Show all executed steps (both completed and failed)
        for i, outcome in enumerate(self.executed_steps, 1):
            tool_name = outcome.action.tool or "N/A"
            status_symbol = "✓" if outcome.status == StepStatus.SUCCESS else "✗"
            lines.append(f"  {i}. {status_symbol} [{tool_name}] {outcome.action.intent}")

        # Show all pending steps
        start_num = len(self.executed_steps) + 1
        for i, step in enumerate(self.pending_steps, start_num):
            tool_name = step.tool or "N/A"
            status = "▶" if i == start_num else "○"
            lines.append(f"  {i}. {status} [{tool_name}] {step.intent}")

        return "\n".join(lines)

    async def execute_current_step(self, tools: dict[str, ContextualTool], believes: list[str],
                                   channel: AgentChannel, context: PursuitContext,
                                   fallback_tool: LLMLogicalOperationTool | None = None) -> 'StepOutcome | None':
        """Advance the plan by executing the next pending action."""
        if self.finished():
            return None

        next_action = self.pending_steps[0]
        await channel.info(f"[{next_action.tool}]: {next_action.intent}")

        # Retrieve context for this specific step using PursuitContext
        step_context = await context.retrieve_context(next_action)
        await channel.info(f"Retrieve {len(step_context)} context for step [{next_action.intent}]")
        await channel.debug(f"Retrieved context for step [{next_action.intent}]: {step_context}")
        # Execute the step with the retrieved context
        outcome = await _execute_step(next_action, tools, believes, step_context, channel, fallback_tool)

        # Store the outcome in context for future steps - regardless of status
        # Key findings should be preserved even from failed or decomposed steps
        await context.store_context(outcome, self.pending_steps[1:])

        return outcome
