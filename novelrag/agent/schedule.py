"""Scheduling system for agent execution."""

from dataclasses import dataclass, field
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from novelrag.agent.channel import AgentChannel
from novelrag.agent.tool import ContextualTool, LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .types import AgentMessageLevel, StepStatus, PursuitStatus, ToolBacklogOutput, ToolConfirmation, ToolMessage, ToolResult, ToolStepDecomposition, ToolStepProgress, ToolUserInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    """Represents a single purposeful action the agent can take.
    
    Steps can spawn sub-steps (decomposition) or trigger
    follow-up steps (chain updates) as the agent discovers
    new requirements.
    """
    tool: str
    intent: str  # What the agent intends to achieve with this action
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Relationships
    contribute_to: 'Step | None' = None  # Step who depends on the results of this step
    spawned_by: 'StepOutcome | None' = None  # Parent action that decomposed into this
    triggered_by: 'StepOutcome | None' = None  # Action that triggered this as follow-up
    
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
                elif isinstance(output, ToolUserInput):
                    if not channel:
                        raise RuntimeError("User input required but no channel provided")
                    user_input = await channel.request(output.prompt)
                elif isinstance(output, ToolResult):
                    outcome.results.append(output.result)
                elif isinstance(output, ToolStepProgress):
                    if output.description:
                        outcome.progress[output.field].append(output.description)
                elif isinstance(output, ToolStepDecomposition):
                    if channel:
                        await channel.message(f"Decomposing into {len(output.steps)} sub-actions")
                    for step in output.steps:
                        sub_action = Step(
                            tool=step['tool'], 
                            intent=step['description'],
                            spawned_by=outcome
                        )
                        outcome.spawned_actions.append(sub_action)
                    outcome.status = StepStatus.DECOMPOSED
                elif isinstance(output, ToolBacklogOutput):
                    outcome.backlog_items.append(output.content)
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
                await channel.message(f"Error: {outcome.error_message}")
        
        outcome.completed_at = datetime.now()
        return outcome


@dataclass
class StepOutcome:
    """The result of executing an action."""
    action: Step
    status: StepStatus
    results: list[str] = field(default_factory=list)
    progress: dict[str, list[str]] = field(default_factory=dict)
    error_message: str | None = None

    # Dynamic plan modifications
    spawned_actions: list[Step] = field(default_factory=list)  # From decomposition
    triggered_actions: list[Step] = field(default_factory=list)  # From chain updates
    
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
    pending_steps: list[Step] = field(default_factory=list)
    completed_steps: list[StepOutcome] = field(default_factory=list)
    failed_steps: list[StepOutcome] = field(default_factory=list)

    def finished(self) -> bool:
        """Check if there are actions waiting to be executed."""
        return len(self.pending_steps) > 0
    
    def build_context(self) -> list[str]:
        """Build context from completed actions for the next action."""
        if self.finished():
            return []
        context = []
        next_action = self.pending_steps[0]
        contributed_actions = [next_action]
        for action_result in self.completed_steps[::-1]:
            if action_result.action.contribute_to in contributed_actions:
                if action_result.status == StepStatus.SUCCESS:
                    context.extend(action_result.results)
                elif action_result.status == StepStatus.FAILED:
                    context.append(json.dumps({
                        "tool": action_result.action.tool,
                        "intent": action_result.action.intent,
                        "status": action_result.status.value,
                        "progress": action_result.progress,
                        "error_message": action_result.error_message
                    }))
                elif action_result.status == StepStatus.DECOMPOSED:
                    context.append(json.dumps({
                        "tool": action_result.action.tool,
                        "intent": action_result.action.intent,
                        "status": action_result.status.value,
                        "spawned_actions": [a.intent for a in action_result.spawned_actions]
                    }))
                contributed_actions.append(action_result.action)
        return context
    
    async def advance(self, tools: dict[str, ContextualTool], believes: list[str] | None = None, 
                      channel: AgentChannel | None = None) -> 'ExecutionPlan':
        """Advance the plan by executing the next pending action."""
        if self.finished():
            return self
        context = self.build_context()
        next_action = self.pending_steps[0]
        outcome = await next_action.execute(tools, believes=believes, context=context, channel=channel)
        # TODO: Reschedule the action based on its outcome
        return ExecutionPlan(
            goal=self.goal,
            pending_steps=self.pending_steps[1:],
            completed_steps=self.completed_steps + [outcome] if outcome.status == StepStatus.SUCCESS else self.completed_steps,
            failed_steps=self.failed_steps + [outcome] if outcome.status == StepStatus.FAILED else self.failed_steps
        )


@dataclass
class GoalPursuitResult:
    """Represents the result of pursuing a specific goal."""
    goal: str
    status: PursuitStatus
    records: ExecutionPlan
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class GoalPursuit:
    """Represents the agent's pursuit of a specific goal."""
    goal: str
    initial_believes: list[str]
    plan: ExecutionPlan
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def new(cls, goal: str, believes: list[str], steps: list[Step]) -> 'GoalPursuit':
        """Create a new goal pursuit instance."""
        plan = ExecutionPlan(
            goal=goal,
            pending_steps=steps,
        )
        return GoalPursuit(
            goal=goal,
            initial_believes=believes,
            plan=plan
        )

    async def advance(self, tools: dict[str, ContextualTool], channel: AgentChannel | None = None) -> 'GoalPursuit | GoalPursuitResult':
        """Execute the goal pursuit."""
        plan = self.plan
        if not plan.finished():
            plan = await self.plan.advance(tools, believes=self.initial_believes, channel=channel)
        
        if plan.finished():
            return GoalPursuitResult(
                goal=self.goal,
                status=PursuitStatus.COMPLETED,
                records=plan,
                started_at=self.started_at,
                completed_at=datetime.now()
            )
        return GoalPursuit(
            goal=self.goal,
            initial_believes=self.initial_believes,
            plan=plan,
            started_at=self.started_at
        )


class GoalPlanner(LLMToolMixin):
    """Responsible for planning and executing goals using the agent's tools."""
    
    def __init__(self, chat_llm: ChatLLM, template_env: TemplateEnvironment):
        super().__init__(chat_llm=chat_llm, template_env=template_env)

    async def plan_goal(self, goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> GoalPursuit:
        """Create an initial execution plan for the given goal."""
        return GoalPursuit(
            goal=goal,
            initial_believes=believes,
            plan=ExecutionPlan(
                goal=goal,
                pending_steps=await self._decompose_goal(goal, believes, tools),
            )
        )
    
    async def _decompose_goal(self, goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[Step]:
        """Decompose a goal into executable steps based on available tools and beliefs."""
        response = await self.call_template(
            "decompose_goal.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools={name: tool.description for name, tool in tools.items()}
        )
        steps = json.loads(response)["steps"]
        steps = [Step(**step) for step in steps]
        return await self._build_step_dependencies(steps)
    
    async def _build_step_dependencies(self, steps: list[Step], target_steps: list[Step] | None = None) -> list[Step]:
        """Build dependency relationships for new steps that will be inserted before existing steps.
        
        Args:
            steps: New steps to be inserted (will have contribute_to relationships built)
            target_steps: Existing steps already in the schedule (new steps will be inserted before these)
            
        Returns:
            Updated list of new steps with contribute_to relationships established
        """
        # Convert steps to serializable format for template
        new_steps = [{"tool": step.tool, "intent": step.intent} for step in steps]
        existing_steps = [{"tool": step.tool, "intent": step.intent} for step in target_steps] if target_steps else []
        
        response = await self.call_template(
            "build_step_dependencies.jinja2",
            json_format=True,
            new_steps=new_steps,
            existing_steps=existing_steps
        )
        dependencies = json.loads(response)["dependencies"]

        # Apply dependencies to create new Step objects
        updated_steps = []
        for i, step in enumerate(steps):
            dependency = next((dep for dep in dependencies if dep["step_index"] == i), None)
            contribute_to = None

            if dependency and dependency["contribute_to"]:
                contribute_ref = dependency["contribute_to"]
                if contribute_ref.startswith("new:"):
                    new_index = int(contribute_ref.split(":")[1])
                    if 0 <= new_index < len(steps):
                        contribute_to = steps[new_index]
                elif contribute_ref.startswith("existing:"):
                    existing_index = int(contribute_ref.split(":")[1])
                    if target_steps and 0 <= existing_index < len(target_steps):
                        contribute_to = target_steps[existing_index]

            # Create new Step with contribute_to relationship
            updated_step = Step(
                tool=step.tool,
                intent=step.intent,
                step_id=step.step_id,
                contribute_to=contribute_to,
                spawned_by=step.spawned_by,
                triggered_by=step.triggered_by
            )
            updated_steps.append(updated_step)
        
        return updated_steps
