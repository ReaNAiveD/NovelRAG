"""Scheduling system for agent execution."""

from dataclasses import dataclass, field
import json
import logging
import uuid
from datetime import datetime

from novelrag.agent.channel import AgentChannel
from novelrag.agent.tool import ContextualTool, LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .types import AgentMessageLevel, StepStatus, PursuitStatus, ToolBacklogOutput, ToolConfirmation, ToolMessage, ToolResult, ToolStepDecomposition, ToolStepProgress, ToolUserInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepDefinition:
    """Represents the core definition of a step - immutable tool and intent description."""
    tool: str
    intent: str  # What the agent intends to achieve with this action
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)


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
    action: ExecutableStep
    status: StepStatus
    results: list[str] = field(default_factory=list)
    progress: dict[str, list[str]] = field(default_factory=dict)    # TODO: progress is used to cache Step status updates for later rerun
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
    
    def build_context(self) -> list[str]:
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
    
    async def advance(self, tools: dict[str, ContextualTool], believes: list[str],
                      channel: AgentChannel, planner: 'GoalPlanner') -> 'ExecutionPlan':
        """Advance the plan by executing the next pending action."""
        if self.finished():
            return self
        context = self.build_context()
        next_action = self.pending_steps[0]
        outcome = await next_action.execute(tools, believes=believes, context=context, channel=channel)
        new_steps = await planner.reschedule_steps(
            goal=self.goal,
            believes=believes,
            tools=tools,
            last_step=outcome,
            target_steps=self.pending_steps[1:],
            prev_steps=self.completed_steps,
        )
        # Handle dependencies of deleted steps
        completed_steps = self.completed_steps + [outcome] if outcome.status == StepStatus.SUCCESS else self.completed_steps
        all_steps = [step.action for step in completed_steps] + new_steps

        # Update contribute_to references for completed steps if their dependencies were deleted
        for step in completed_steps:
            if step.action.contribute_to and step.action.contribute_to not in all_steps:
                # Traverse the contribute_to chain until we find a step that exists in the current plan
                current_target = step.action.contribute_to
                while current_target and current_target not in all_steps:
                    current_target = current_target.contribute_to
                step.action.contribute_to = current_target

        return ExecutionPlan(
            goal=self.goal,
            pending_steps=new_steps,
            completed_steps=completed_steps,
            failed_steps=self.failed_steps + [outcome] if outcome.status != StepStatus.SUCCESS else self.failed_steps
        )


@dataclass(frozen=True)
class GoalPursuitResult:
    """Represents the result of pursuing a specific goal."""
    goal: str
    status: PursuitStatus
    records: ExecutionPlan
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class GoalPursuit:
    """Represents the agent's pursuit of a specific goal."""
    goal: str
    initial_believes: list[str]
    plan: ExecutionPlan
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def new(cls, goal: str, believes: list[str], steps: list[ExecutableStep]) -> 'GoalPursuit':
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

    async def advance(self, tools: dict[str, ContextualTool], channel: AgentChannel, planner: 'GoalPlanner') -> 'GoalPursuit | GoalPursuitResult':
        """Execute the goal pursuit."""
        plan = self.plan
        if not plan.finished():
            plan = await self.plan.advance(tools, believes=self.initial_believes, channel=channel, planner=planner)

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
        """
        Create an initial execution plan for the given goal.
        This method decomposes the goal into actionable steps
        and initializes a GoalPursuit instance.
        Args:
            goal: The goal to pursue.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.
        Returns:
            A GoalPursuit instance containing the initial plan.
        """
        return GoalPursuit(
            goal=goal,
            initial_believes=believes,
            plan=ExecutionPlan(
                goal=goal,
                pending_steps=await self._decompose_goal(goal, believes, tools),
            )
        )

    async def reschedule_steps(
            self, last_step: StepOutcome, target_steps: list[ExecutableStep], prev_steps: list[StepOutcome],
            goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Reschedule steps based on the last completed step and current context.

        This method dynamically updates the execution plan by analyzing the outcome
        of the most recently completed step. It can create new steps, remove obsolete
        ones, or modify existing steps to adapt to changing circumstances. The method
        ensures proper dependency relationships are maintained between new and existing steps.

        The rescheduling process considers:
        - Success/failure status and results of the last completed step
        - Current agent beliefs and available tools
        - Existing steps that may no longer be relevant
        - New steps that may be needed based on discovered information

        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            target_steps: The list of pending steps in the current schedule that
                         may need to be updated, removed, or supplemented.
            prev_steps: All previously completed step outcomes that provide
                       historical context for rescheduling decisions.
            goal: The overall goal the agent is pursuing, used to ensure
                 rescheduled steps remain aligned with the objective.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.

        Returns:
            A list of updated Step instances representing the new schedule,
            with proper dependency relationships established between steps.

        Note:
            This method is called internally during plan execution when the
            agent needs to adapt its strategy based on new information or
            changing circumstances.
        """
        prev_step_dicts = [{
            "tool": prev_step.action.tool,
            "intent": prev_step.action.intent,
            "status": prev_step.status.value,
            "results": prev_step.results
        } for prev_step in prev_steps]
        last_step_dict = {
            "tool": last_step.action.tool,
            "intent": last_step.action.intent,
            "status": last_step.status.value,
            "results": last_step.results,
            "progress": last_step.progress,
            "error_message": last_step.error_message
        }
        spawned_steps = last_step.spawned_actions
        triggered_steps = last_step.triggered_actions
        spawned_step_dicts = [{"tool": step.tool, "intent": step.intent} for step in spawned_steps]
        triggered_step_dicts = [{"tool": step.tool, "intent": step.intent} for step in triggered_steps]
        target_step_dicts = [{"tool": step.tool, "intent": step.intent} for step in target_steps]
        tools = {name: tool.description for name, tool in tools.items()}
        response = await self.call_template(
            "reschedule_steps.jinja2",
            json_format=True,
            last_step=last_step_dict,
            target_steps=target_step_dicts,
            prev_steps=prev_step_dicts,
            spawned_steps=spawned_step_dicts,
            triggered_steps=triggered_step_dicts,
            goal=goal,
            believes=believes,
            tools=tools
        )
        response = json.loads(response)
        new_steps = response["new_steps"]
        delete_count = response.get("delete_count", 0)
        steps = [ExecutableStep(definition=StepDefinition(**step)) for step in new_steps]
        target_steps = target_steps[delete_count:] if delete_count < len(target_steps) else []
        steps = await self._build_step_dependencies(steps, target_steps)
        return spawned_steps + ([last_step.action] if last_step.status != StepStatus.SUCCESS else triggered_steps) + steps + target_steps

    async def _decompose_goal(self, goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """Decompose a goal into executable steps based on available tools and beliefs."""
        response = await self.call_template(
            "decompose_goal.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools={name: tool.description for name, tool in tools.items()}
        )
        steps = json.loads(response)["steps"]
        steps = [ExecutableStep(definition=StepDefinition(**step)) for step in steps]
        return await self._build_step_dependencies(steps)
    
    async def _build_step_dependencies(self, steps: list[ExecutableStep], target_steps: list[ExecutableStep] | None = None) -> list[ExecutableStep]:
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
            updated_step = ExecutableStep(
                definition=StepDefinition(
                    tool=step.tool,
                    intent=step.intent,
                    step_id=step.step_id,
                ),
                contribute_to=contribute_to,
                spawned_by=step.spawned_by,
                triggered_by=step.triggered_by
            )
            updated_steps.append(updated_step)
        
        return updated_steps
