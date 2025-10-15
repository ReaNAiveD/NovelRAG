import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .channel import AgentChannel
from .workspace import ResourceContext
from .execution import ExecutionPlan
from .planning import PursuitPlanner
from .steps import StepDefinition, StepStatus, StepOutcome
from .tool import ContextualTool, LLMToolMixin, LLMLogicalOperationTool
from ..llm import ChatLLM
from ..template import TemplateEnvironment


class PursuitStatus(Enum):
    """Status of a goal pursuit."""
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class GoalPursuitResult:
    """Represents the result of pursuing a specific goal."""
    goal: str
    status: PursuitStatus
    records: ExecutionPlan
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass()
class GoalPursuit:
    """Represents the agent's pursuit of a specific goal."""
    goal: str
    initial_believes: list[str]
    plan: ExecutionPlan
    context: ResourceContext
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def new(cls, goal: str, believes: list[str], steps: list[StepDefinition], context: 'ResourceContext') -> 'GoalPursuit':
        """Create a new goal pursuit instance."""
        plan = ExecutionPlan(
            goal=goal,
            pending_steps=steps,
        )
        return GoalPursuit(
            goal=goal,
            initial_believes=believes,
            plan=plan,
            context=context
        )

    @classmethod
    async def initialize_pursuit(
            cls,
            goal: str,
            believes: list[str],
            planner: PursuitPlanner,
            tools: dict[str, ContextualTool],
            context: 'ResourceContext',
    ):
        """Initialize a goal pursuit with a plan based on the goal and beliefs."""
        steps = await planner.create_initial_plan(goal, believes, tools, context)
        return cls.new(goal, believes, steps, context)

    async def execute_next_step(self, tools: dict[str, ContextualTool], channel: AgentChannel, planner: 'PursuitPlanner', fallback_tool: LLMLogicalOperationTool | None = None) -> 'GoalPursuitResult | None':
        """Execute the goal pursuit."""
        plan = self.plan
        if not plan.finished():
            outcome = await self.plan.execute_current_step(tools, believes=self.initial_believes, channel=channel, context=self.context, fallback_tool=fallback_tool)
            if not outcome:
                await channel.error(f"The plan is not ready to execute the next step.")
                return None
            await channel.info(f"Executed step [{outcome.status.value}]: {outcome.action.intent}")
            if outcome.status == StepStatus.FAILED:
                await channel.error(f"Step[{outcome.action.tool}] failed: {outcome.error_message}")
            elif outcome.status == StepStatus.SUCCESS:
                await channel.info(f"Step results: {outcome.results}")

            new_steps = await planner.adapt_plan(
                last_step=outcome,
                original_plan=self.plan,
                believes=self.initial_believes,
                tools=tools,
                context=self.context
            )
            # Update the plan with new steps
            executed_steps = self.plan.executed_steps + [outcome]
            new_plan = ExecutionPlan(
                goal=self.goal,
                pending_steps=new_steps,
                executed_steps=executed_steps
            )
            self.plan = new_plan
            await channel.info(f"New plan for goal '{self.goal}': {new_plan}")

        if plan.finished():
            return GoalPursuitResult(
                goal=self.goal,
                status=PursuitStatus.COMPLETED,
                records=plan,
                started_at=self.started_at,
                completed_at=datetime.now()
            )
        return None

    async def run_to_completion(self, tools: dict[str, ContextualTool], channel: AgentChannel, planner: 'PursuitPlanner', fallback_tool: LLMLogicalOperationTool | None = None) -> GoalPursuitResult:
        """Run the goal pursuit until completion."""
        while True:
            result = await self.execute_next_step(tools, channel, planner, fallback_tool)
            if result is not None:
                return result


class PursuitSummarizer(LLMToolMixin):
    """Responsible for summarizing the results of a goal pursuit."""

    def __init__(self, chat_llm: ChatLLM, template_env: TemplateEnvironment):
        super().__init__(chat_llm=chat_llm, template_env=template_env)

    async def summarize_pursuit(self, pursuit: GoalPursuitResult) -> str:
        """Generate a summary of the goal pursuit results."""
        key_steps = await self._identify_key_steps(pursuit)
        summary = await self._generate_summary(pursuit, key_steps)
        return summary

    async def _identify_key_steps(self, pursuit: GoalPursuitResult, threshold: int = 3) -> list[StepOutcome]:
        """Identify key steps in the pursuit based on their impact or significance."""
        steps = [{"tool": step.action.tool, "intent": step.action.intent, "status": step.status.value} for step in pursuit.records.executed_steps]
        response = await self.call_template(
            "identify_key_steps.jinja2",
            json_format=True,
            step_goal=pursuit.goal,
            pursuit_status=pursuit.status.value,
            steps=steps,
            threshold=threshold
        )
        key_step_indices = json.loads(response)['indices']
        key_steps = [pursuit.records.executed_steps[i] for i in key_step_indices if i < len(pursuit.records.executed_steps)]
        return key_steps

    async def _generate_summary(self, pursuit: GoalPursuitResult, key_steps: list[StepOutcome]) -> str:
        """Generate a detailed summary of the pursuit including key steps."""
        steps = [{"tool": step.action.tool, "intent": step.action.intent, "status": step.status.value} for step in pursuit.records.executed_steps]
        key_step_results = [{"tool": step.action.tool, "intent": step.action.intent, "results": step.results} for step in key_steps]
        response = await self.call_template(
            "generate_pursuit_summary.jinja2",
            goal=pursuit.goal,
            status=pursuit.status.value,
            steps=steps,
            key_steps=key_step_results,
        )
        return response
