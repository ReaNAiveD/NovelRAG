import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .channel import AgentChannel
from .tool import ContextualTool
from .types import ToolMessage, ToolConfirmation, ToolUserInput, ToolResult, ToolStepProgress, ToolStepDecomposition, ToolBacklogOutput, AgentMessageLevel

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of an action's execution."""
    SUCCESS = "success"
    FAILED = "failed"
    DECOMPOSED = "decomposed"  # Action was broken down into sub-actions
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class StepDefinition:
    """Represents the core definition of a step - immutable tool and intent description."""
    tool: str
    intent: str  # What the agent intends to achieve with this action
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    progress: dict[str, list[str]] = field(default_factory=dict)  # Progress for recovery and tracking


@dataclass
class ExecutableStep:
    """Represents an executable step with relationships and execution capability."""
    definition: StepDefinition

    # Relationships
    contribute_to: 'ExecutableStep | None' = None  # Step who depends on the results of this step
    spawned_by: 'StepOutcome | None' = None  # Parent action that decomposed into this
    triggered_by: 'StepOutcome | None' = None  # Action that triggered this as follow-up

    @property
    def tool(self) -> str:
        """Access to the tool from the definition."""
        return self.definition.tool

    @property
    def intent(self) -> str:
        """Access to the intent from the definition."""
        return self.definition.intent

    @property
    def step_id(self) -> str:
        """Access to the step_id from the definition."""
        return self.definition.step_id

    async def execute(self, tools: dict[str, ContextualTool], believes: list[str] | None = None,
                      context: list[str] | None = None, channel: AgentChannel | None = None) -> 'StepOutcome':
        """Execute the action and return its outcome."""
        start_time = datetime.now()
        outcome = StepOutcome(
            action=self,
            status=StepStatus.FAILED,
            started_at=start_time
        )

        # Validate tool exists
        tool = tools.get(self.tool)
        if not tool:
            outcome.error_message = f"Tool {self.tool} not found."
            outcome.completed_at = datetime.now()
            if channel:
                await channel.message(outcome.error_message)
            return outcome

        try:
            # Initialize tool call
            tool_call = tool.call(
                believes=believes,
                step_description=self.intent,
                context=context,
                tools={name: t.description or '' for name, t in tools.items()}
            )

            # Process tool outputs
            user_input = None
            while True:
                output = await tool_call.asend(user_input)
                user_input = None  # Reset for next iteration

                if isinstance(output, ToolMessage):
                    if channel:
                        level = AgentMessageLevel(output.level.value)
                        await channel.send_message(output.content, level=level)
                elif isinstance(output, ToolConfirmation):
                    if not channel:
                        raise RuntimeError("Confirmation required but no channel provided")
                    user_input = await channel.confirm(output.prompt)
                    await channel.debug(f"User confirmed: {user_input}")
                elif isinstance(output, ToolUserInput):
                    if not channel:
                        raise RuntimeError("User input required but no channel provided")
                    user_input = await channel.request(output.prompt)
                    await channel.debug(f"User input received: {user_input}")
                elif isinstance(output, ToolResult):
                    outcome.results.append(output.result)
                elif isinstance(output, ToolStepProgress):
                    outcome.progress[output.field].append(output.description)
                elif isinstance(output, ToolStepDecomposition):
                    if channel:
                        await channel.message(f"Decomposing into {len(output.steps)} sub-actions")
                    for step in output.steps:
                        sub_action = ExecutableStep(
                            definition=StepDefinition(
                                tool=step['tool'],
                                intent=step['description']
                            ),
                            spawned_by=outcome,
                            contribute_to=self,
                        )
                        outcome.spawned_actions.append(sub_action)
                    outcome.status = StepStatus.DECOMPOSED
                elif isinstance(output, ToolBacklogOutput):
                    outcome.backlog_items.append(output.content)
                    await channel.debug(f"Added to backlog: {output.content}")
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")
        except StopAsyncIteration:
            # Tool completed successfully
            outcome.status = StepStatus.SUCCESS if outcome.status != StepStatus.DECOMPOSED else outcome.status
            if channel:
                await channel.message(f"Action completed: {outcome.status.value}")

        except Exception as e:
            outcome.status = StepStatus.FAILED
            outcome.error_message = str(e)
            if channel:
                await channel.error(f"Error: {outcome.error_message}")

        outcome.completed_at = datetime.now()
        return outcome


@dataclass
class StepOutcome:
    """The result of executing an action."""
    action: ExecutableStep
    status: StepStatus
    results: list[str] = field(default_factory=list)
    progress: dict[str, list[str]] = field(default_factory=dict)
    error_message: str | None = None

    # Dynamic plan modifications
    spawned_actions: list[ExecutableStep] = field(default_factory=list)  # From decomposition
    triggered_actions: list[ExecutableStep] = field(default_factory=list)  # From chain updates

    # Discovered insights and future work
    discovered_insights: list[str] = field(default_factory=list)
    backlog_items: list[str] = field(default_factory=list)

    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Represents the agent's evolving plan to achieve a goal.

    Unlike a schedule which implies fixed timing, an execution plan
    adapts dynamically as new information emerges.
    """
    goal: str
    pending_steps: list[ExecutableStep] = field(default_factory=list)
    completed_steps: list[StepOutcome] = field(default_factory=list)
    failed_steps: list[StepOutcome] = field(default_factory=list)

    def finished(self) -> bool:
        """Check if there are actions waiting to be executed."""
        return len(self.pending_steps) == 0

    def gather_execution_context(self) -> list[str]:
        """Build context from completed actions for the next action."""
        if self.finished():
            return []
        context = []
        next_action = self.pending_steps[0]
        contributed_actions = [next_action]
        for action_result in self.completed_steps[::-1]:
            if action_result.action.contribute_to not in contributed_actions:
                continue
            if action_result.status == StepStatus.SUCCESS:
                context.extend(action_result.results)
            elif action_result.status == StepStatus.FAILED:
                context.append(json.dumps({
                    "tool": action_result.action.tool,
                    "intent": action_result.action.intent,
                    "status": action_result.status.value,
                    "progress": action_result.progress,
                    "error_message": action_result.error_message
                }, ensure_ascii=False))
            elif action_result.status == StepStatus.DECOMPOSED:
                context.append(json.dumps({
                    "tool": action_result.action.tool,
                    "intent": action_result.action.intent,
                    "status": action_result.status.value,
                    "spawned_actions": [a.intent for a in action_result.spawned_actions]
                }, ensure_ascii=False))
            contributed_actions.append(action_result.action)
        return context

    async def execute_current_step(self, tools: dict[str, ContextualTool], believes: list[str],
                                   channel: AgentChannel) -> 'StepOutcome | None':
        """Advance the plan by executing the next pending action."""
        if self.finished():
            return None
        context = self.gather_execution_context()
        next_action = self.pending_steps[0]
        await channel.debug(f"Executing action: {next_action.intent} with tool {next_action.tool}")
        await channel.debug(f"Context for action: {context}")
        outcome = await next_action.execute(tools, believes=believes, context=context, channel=channel)
        await channel.debug(f"Action outcome: {outcome.status.value} with results: {outcome.results}")
        return outcome
