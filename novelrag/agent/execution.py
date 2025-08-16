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
    def __init__(self, channel: AgentChannel | None = None, backlog: list[str] | None = None, progress: dict[str, list[str]] | None = None):
        self.channel = channel
        self._backlog = backlog
        self._progress = progress

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

    async def progress(self, key: str, value: Any, description: str | None = None):
        if self._progress is not None:
            if key not in self._progress:
                self._progress[key] = []
            self._progress[key].append(value)

    async def backlog(self, content: str, priority: str | None = None):
        if self._backlog is not None:
            self._backlog.append(content)


# Execute step function - works directly with StepDefinition
async def _execute_step(step: StepDefinition, tools: dict[str, ContextualTool], believes: list[str] | None = None,
                        context: list[str] | None = None, channel: AgentChannel | None = None,
                        fallback_tool: LLMLogicalOperationTool | None = None) -> StepOutcome:
    """Execute the action and return its outcome."""
    start_time = datetime.now()
    outcome = StepOutcome(
        action=step,
        status=StepStatus.SUCCESS,
        started_at=start_time
    )

    # Get the tool, use fallback if tool is None or not found
    tool = None
    if step.tool and step.tool in tools:
        tool = tools[step.tool]
    elif step.tool is None or step.tool == '' or step.tool not in tools:
        if fallback_tool:
            tool = fallback_tool
            if channel:
                await channel.debug(f"Using LLMLogicalOperationTool for step with tool='{step.tool}' - performing logical operation: {step.intent}")
        else:
            outcome.status = StepStatus.FAILED
            outcome.error_message = f"Tool {step.tool} not found and no fallback tool available."
            outcome.completed_at = datetime.now()
            return outcome

    if not tool:
        outcome.status = StepStatus.FAILED
        outcome.error_message = f"Tool {step.tool} not found."
        outcome.completed_at = datetime.now()
        return outcome

    try:
        # Initialize tool call
        runtime = ExecutionToolRuntime(
            channel=channel,
            backlog=outcome.backlog_items,
            progress=outcome.progress
        )
        result = await tool.call(
            runtime=runtime,
            believes=believes,
            step_description=step.intent,
            context=context,
            tools={name: t.description or '' for name, t in tools.items()}
        )

        if isinstance(result, ToolDecomposition):
            # Handle tool decomposition into sub-actions
            outcome.status = StepStatus.DECOMPOSED
            outcome.progress = result.progress
            for sub_step in result.steps:
                sub_action = StepDefinition(
                    tool=sub_step['tool'],
                    intent=sub_step['description'],
                    progress=result.progress,
                    reason="decomposed",
                    reason_details=f"Decomposed from: {step.intent}"
                )
                outcome.decomposed_actions.append(sub_action)
        elif isinstance(result, ToolResult):
            # Handle tool result
            outcome.status = StepStatus.SUCCESS
            outcome.results.append(result.result)
        elif isinstance(result, ToolError):
            outcome.status = StepStatus.FAILED
            outcome.error_message = result.error_message
        else:
            raise ValueError(f"Unexpected tool output type: {type(result)}")
    except Exception as e:
        outcome.status = StepStatus.FAILED
        outcome.error_message = str(e)

    outcome.completed_at = datetime.now()
    return outcome


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
        await channel.info(f"Retrieved context for step {next_action.step_id}: {step_context}")
        # Execute the step with the retrieved context
        outcome = await _execute_step(next_action, tools, believes, step_context, channel, fallback_tool)

        # Store the outcome in context for future steps
        if outcome.status == StepStatus.SUCCESS:
            await context.store_context(outcome, self.pending_steps[1:])

        return outcome
