import json
import logging

from .execution import ExecutionPlan
from .steps import StepDefinition, StepOutcome, StepStatus
from .tool import ContextualTool, LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class PursuitPlanner(LLMToolMixin):
    """Responsible for planning and executing goals using the agent's tools."""
    
    def __init__(self, chat_llm: ChatLLM, template_env: TemplateEnvironment):
        super().__init__(chat_llm=chat_llm, template_env=template_env)

    async def create_initial_plan(self, goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Decompose a goal into executable steps based on available tools and beliefs.
        Args:
            goal: The goal to pursue.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.
        Returns:
            A list of StepDefinition instances representing the decomposed steps.
        """
        response = await self.call_template(
            "decompose_goal.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools={name: tool.description for name, tool in tools.items()}
        )
        steps = json.loads(response)["steps"]
        steps = [StepDefinition(**step, reason="initial_plan") for step in steps]
        return steps

    async def adapt_plan(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Adapt the execution plan based on the last completed step and current context.
        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances

        Returns:
            A list of updated StepDefinition instances representing the new schedule,
            incorporating any necessary adjustments based on the last step's outcome.
        """
        if last_step.status == StepStatus.SUCCESS:
            return await self._adapt_plan_from_success(last_step, original_plan, believes, tools)
        elif last_step.status == StepStatus.FAILED:
            return await self._adapt_plan_from_failure(last_step, original_plan, believes, tools)
        elif last_step.status == StepStatus.DECOMPOSED:
            return await self._adapt_plan_from_decomposition(last_step, original_plan, believes, tools)
        else:
            logging.warning(f"Unhandled step status: {last_step.status}. Returning remaining steps.")
            return original_plan.pending_steps[1:]

    async def _adapt_plan_from_success(self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str], tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Adapt the plan after a successful step execution.

        Args:
            last_step: The successfully completed step outcome.
            original_plan: The original execution plan.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.

        Returns:
            A list of updated StepDefinition instances representing the new schedule,
            potentially including triggered follow-up actions based on the success.
        """

        remaining_steps, triggered_steps = await self._plan_remaining_steps(
            [last_step], original_plan.pending_steps[1:], believes, tools
        )

        # Add triggered steps with appropriate reason
        for step in triggered_steps:
            step = StepDefinition(
                intent=step.intent,
                tool=step.tool,
                progress=step.progress,
                reason="triggered",
                reason_details=f"Triggered by successful completion of: {last_step.action.intent}"
            )

        return remaining_steps + triggered_steps

    async def _adapt_plan_from_failure(self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str], tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Adapt the plan after a failed step execution.

        Args:
            last_step: The failed step outcome.
            original_plan: The original execution plan.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.

        Returns:
            A list of updated StepDefinition instances representing the new schedule,
            potentially including recovery or alternative steps.
        """

        remaining_steps, recovery_steps = await self._plan_remaining_steps(
            [last_step], original_plan.pending_steps[1:], believes, tools
        )

        # Add recovery steps with appropriate reason
        for step in recovery_steps:
            step = StepDefinition(
                intent=step.intent,
                tool=step.tool,
                progress=step.progress,
                reason="recovery",
                reason_details=f"Recovery from failed step: {last_step.action.intent} - {last_step.error_message}"
            )

        return recovery_steps + remaining_steps

    async def _adapt_plan_from_decomposition(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Adapt the plan after a step was decomposed into sub-actions.

        Args:
            last_step: The decomposed step outcome containing spawned actions.
            original_plan: The original execution plan.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.

        Returns:
            A list of updated StepDefinition instances representing the new schedule,
            with spawned actions inserted appropriately.
        """

        # The spawned actions already have proper reason fields set in execution
        spawned_actions = last_step.decomposed_actions
        remaining_steps = original_plan.pending_steps[1:]

        return spawned_actions + remaining_steps

    async def _plan_remaining_steps(self, executed_steps: list[StepOutcome], planned_steps: list[StepDefinition],
                                    believes: list[str], tools: dict[str, ContextualTool]) -> tuple[list[StepDefinition], list[StepDefinition]]:
        """
        Plan the remaining steps based on executed outcomes and current plan.
        Args:
            executed_steps: List of StepOutcome instances representing the steps that have already been executed.
            planned_steps: List of StepDefinition instances representing the steps that are planned but not yet executed.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.
        Returns:
            A tuple of (remaining_steps, additional_steps) where remaining_steps are the
            updated planned steps and additional_steps are any new steps to be added.
        """

        response = await self.call_template(
            "plan_remaining_steps.jinja2",
            json_format=True,
            executed_steps=[{
                "intent": outcome.action.intent,
                "tool": outcome.action.tool,
                "status": outcome.status.value,
                "results": outcome.results,
                "error_message": outcome.error_message
            } for outcome in executed_steps],
            planned_steps=[{
                "intent": step.intent,
                "tool": step.tool
            } for step in planned_steps],
            believes=believes,
            tools={name: tool.description for name, tool in tools.items()}
        )

        result = json.loads(response)
        remaining_steps = [StepDefinition(**step) for step in result.get("remaining_steps", [])]
        additional_steps = [StepDefinition(**step) for step in result.get("additional_steps", [])]

        return remaining_steps, additional_steps
