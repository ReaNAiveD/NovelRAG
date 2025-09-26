import json
import logging

from novelrag.agent.plan_strategy import NoOpPlanningStrategy, PlanningStrategy

from .execution import ExecutionPlan
from .context import PursuitContext
from .steps import StepDefinition, StepOutcome, StepStatus
from .tool import ContextualTool, LLMToolMixin, SchematicToolAdapter
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class PursuitPlanner(LLMToolMixin):
    """Responsible for planning and executing goals using the agent's tools."""
    
    def __init__(self, chat_llm: ChatLLM, template_env: TemplateEnvironment, strategy: PlanningStrategy | None = None):
        super().__init__(chat_llm=chat_llm, template_env=template_env)
        self.strategy = strategy or NoOpPlanningStrategy()

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
        # Build tool info including input schemas for SchematicTools
        tool_info = {}
        for name, tool in tools.items():
            info: dict[str, str | dict | None] = {"description": tool.description}
            if isinstance(tool, SchematicToolAdapter):
                info["input_schema"] = tool.inner.input_schema
            tool_info[name] = info

        planning_instructions = self.strategy.initial_planning_instructions()

        response = await self.call_template(
            "create_initial_plan.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools=tool_info,
            planning_instructions=planning_instructions
        )
        steps = json.loads(response)["steps"]
        steps = [StepDefinition(**step) for step in steps]
        steps = self.strategy.post_planning(steps)
        return steps

    async def adapt_plan(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool], context: PursuitContext) -> list[StepDefinition]:
        """
        Adapt the execution plan based on the last completed step and current context.

        The new replan logic:
        1. Retrieves relevant context for planning decisions based on goal, last step, and original plan
        2. Creates a new plan for future work based on ALL executed steps and insights
        3. Compares the new plan with original pending steps to identify insights and gaps
        4. Merges plans intelligently based on whether there are significant new insights
        5. Handles different step statuses with appropriate priority steps (triggered/decomposed)

        Args:
            last_step: The outcome of the most recently completed step
            original_plan: The original execution plan with executed and pending steps
            believes: The agent's current beliefs about the world state
            tools: Dictionary mapping tool names to ContextualTool instances
            context: PursuitContext for retrieving relevant historical information

        Returns:
            A list of updated StepDefinition instances representing the new schedule
        """
        try:
            # Retrieve relevant context for planning decisions
            planning_context = await context.retrieve_planning_context(
                goal=original_plan.goal,
                last_step=last_step,
                completed_steps=original_plan.executed_steps,
                pending_steps=original_plan.pending_steps
            )
            planning_context = self.strategy.filter_planning_context(planning_context)

            if last_step.status == StepStatus.SUCCESS:
                return await self._adapt_plan_from_success(last_step, original_plan, believes, tools, planning_context)
            elif last_step.status == StepStatus.FAILED:
                return await self._adapt_plan_from_failure(last_step, original_plan, believes, tools, planning_context)
            elif last_step.status == StepStatus.DECOMPOSED:
                return await self._adapt_plan_from_decomposition(last_step, original_plan, believes, tools, planning_context)
            else:
                logging.warning(f"Unhandled step status: {last_step.status}. Using simple fallback.")
                return original_plan.pending_steps[1:]
        except Exception as e:
            logging.error(f"Error in adapt_plan: {e}. Using fallback.")
            return original_plan.pending_steps[1:]

    async def _adapt_plan_from_success(self, last_step: StepOutcome, original_plan: ExecutionPlan,
                                     believes: list[str], tools: dict[str, ContextualTool],
                                     planning_context: list[str]) -> list[StepDefinition]:
        """
        Adapt plan after successful step execution.
        Creates unified execution plan with immediate steps (triggered) first.
        """
        # Get triggered steps to go first
        immediate_steps = []
        if last_step.triggered_actions:
            immediate_steps = await self._convert_triggered_actions_to_steps(
                last_step, tools, believes, planning_context
            )

        # Create unified execution plan
        execution_plan = await self._create_future_plan(
            original_plan.executed_steps + [last_step], original_plan.goal, believes, tools, planning_context,
            immediate_steps, original_plan.pending_steps[1:]
        )

        return self.strategy.post_planning(execution_plan)

    async def _adapt_plan_from_failure(self, last_step: StepOutcome, original_plan: ExecutionPlan,
                                     believes: list[str], tools: dict[str, ContextualTool],
                                     planning_context: list[str]) -> list[StepDefinition]:
        """
        Adapt plan after failed step execution.
        Creates unified execution plan with immediate steps (recovery + optional rerun) first.
        """
        # Analyze failure with full execution history
        failure_analysis = await self._analyze_failure(
            last_step, original_plan.executed_steps, original_plan.goal, believes, tools, planning_context
        )

        # Build immediate steps (recovery + optional rerun)
        immediate_steps = []

        # Add recovery steps
        for step_data in failure_analysis.get("recovery_steps", []):
            immediate_steps.append(StepDefinition(**step_data))

        # Add rerun if recommended
        if failure_analysis.get("should_rerun", False):
            immediate_steps.append(StepDefinition(
                intent=last_step.action.intent,
                tool=last_step.action.tool,
                progress=last_step.action.progress,
                reason="rerun",
                reason_details=failure_analysis.get("rerun_reason", "Rerun after recovery steps")
            ))

        # Create unified execution plan with recovery context
        execution_plan = await self._create_future_plan(
            original_plan.executed_steps + [last_step], original_plan.goal, believes, tools, planning_context,
            immediate_steps, original_plan.pending_steps[1:]
        )

        return self.strategy.post_planning(execution_plan)

    async def _adapt_plan_from_decomposition(self, last_step: StepOutcome, original_plan: ExecutionPlan,
                                           believes: list[str], tools: dict[str, ContextualTool],
                                           planning_context: list[str]) -> list[StepDefinition]:
        """
        Adapt plan after step decomposition.
        Creates unified execution plan with immediate steps (decomposed) first.
        """
        # Build immediate steps (only decomposed steps)
        immediate_steps: list[StepDefinition] = []

        # Add decomposed steps
        if last_step.decomposed_actions:
            immediate_steps = await self._convert_decomposed_actions_to_steps(
                last_step, tools, believes, planning_context
            )

        # Create unified execution plan
        execution_plan = await self._create_future_plan(
            original_plan.executed_steps + [last_step], original_plan.goal, believes, tools, planning_context,
            immediate_steps, original_plan.pending_steps[1:]
        )

        return self.strategy.post_planning(execution_plan)

    async def _create_future_plan(self, executed_steps: list[StepOutcome], goal: str,
                                believes: list[str], tools: dict[str, ContextualTool],
                                planning_context: list[str], immediate_steps: list[StepDefinition],
                                original_pending_steps: list[StepDefinition]) -> list[StepDefinition]:
        """
        Create a complete execution plan by merging immediate steps with updated original steps.

        Args:
            executed_steps: All steps executed so far, including the last step
            goal: The original goal being pursued
            believes: Current beliefs of the agent
            tools: Available tools
            planning_context: Relevant historical context for planning decisions
            immediate_steps: Priority steps that must execute first (triggered/decomposed/recovery)
            original_pending_steps: Original remaining steps from the plan

        Returns:
            List of StepDefinition instances representing the complete execution plan
        """
        if not executed_steps:
            return []

        last_step = executed_steps[-1]

        # Build tool info including input schemas for SchematicTools
        tool_info = {}
        for name, tool in tools.items():
            info: dict[str, str | dict | None] = {"description": tool.description}
            if isinstance(tool, SchematicToolAdapter):
                info["input_schema"] = tool.inner.input_schema
            tool_info[name] = info

        adapt_instructions = self.strategy.adapt_planning_instructions()

        response = await self.call_template(
            "create_future_plan.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools=tool_info,
            planning_context=planning_context,
            adapt_planning_instructions=adapt_instructions,
            immediate_steps=[{
                "intent": step.intent,
                "tool": step.tool,
                "reason": step.reason,
                "reason_details": step.reason_details
            } for step in immediate_steps],
            original_pending_steps=[{
                "intent": step.intent,
                "tool": step.tool,
                "reason": step.reason,
                "reason_details": step.reason_details
            } for step in original_pending_steps],
            executed_steps=[{
                "intent": outcome.action.intent,
                "tool": outcome.action.tool,
                "status": outcome.status.value,
                "results": outcome.results,
                "error_message": outcome.error_message,
                "discovered_insights": outcome.discovered_insights
            } for outcome in executed_steps],
            last_step={
                "intent": last_step.action.intent,
                "tool": last_step.action.tool,
                "status": last_step.status.value,
                "results": last_step.results,
                "error_message": last_step.error_message,
                "discovered_insights": last_step.discovered_insights,
                "triggered_actions": last_step.triggered_actions,
                "decomposed_actions": last_step.decomposed_actions
            }
        )

        result = json.loads(response)
        execution_plan = [StepDefinition(**step) for step in result.get("steps", [])]
        return execution_plan

    async def _convert_triggered_actions_to_steps(
        self,
        last_step: StepOutcome,
        tools: dict[str, ContextualTool],
        believes: list[str],
        planning_context: list[str]
    ) -> list[StepDefinition]:
        """
        Convert triggered actions from dict format to StepDefinition objects.
        
        Args:
            last_step: The step outcome containing triggered actions
            tools: Available tools for mapping
            believes: Current beliefs of the agent
            planning_context: Relevant historical context
            
        Returns:
            List of StepDefinition objects converted from triggered actions
        """
        if not last_step.triggered_actions:
            return []

        # Build tool info including input schemas for SchematicTools
        tool_info = {}
        for name, tool in tools.items():
            info: dict[str, str | dict | None] = {"description": tool.description}
            if isinstance(tool, SchematicToolAdapter):
                info["input_schema"] = tool.inner.input_schema
            tool_info[name] = info

        response = await self.call_template(
            "convert_triggered_actions.jinja2",
            json_format=True,
            triggered_actions=last_step.triggered_actions,
            triggering_step_intent=last_step.action.intent,
            tools=tool_info,
            believes=believes,
            planning_context=planning_context
        )

        result = json.loads(response)
        triggered_steps = []
        
        for step_data in result.get("steps", []):
            triggered_steps.append(StepDefinition(
                intent=step_data["intent"],
                tool=step_data["tool"],
                reason="triggered",
                reason_details=f"Triggered by successful completion of: {last_step.action.intent}"
            ))

        return triggered_steps

    async def _convert_decomposed_actions_to_steps(
        self,
        last_step: StepOutcome,
        tools: dict[str, ContextualTool],
        believes: list[str],
        planning_context: list[str]
    ) -> list[StepDefinition]:
        """
        Convert decomposed actions from dict format to StepDefinition objects.
        
        Args:
            last_step: The step outcome containing decomposed actions
            tools: Available tools for mapping
            believes: Current beliefs of the agent
            planning_context: Relevant historical context
            
        Returns:
            List of StepDefinition objects converted from decomposed actions
        """
        if not last_step.decomposed_actions:
            return []

        # Build tool info including input schemas for SchematicTools
        tool_info = {}
        for name, tool in tools.items():
            info: dict[str, str | dict | None] = {"description": tool.description}
            if isinstance(tool, SchematicToolAdapter):
                info["input_schema"] = tool.inner.input_schema
            tool_info[name] = info

        response = await self.call_template(
            "convert_decomposed_actions.jinja2",
            json_format=True,
            decomposed_actions=last_step.decomposed_actions,
            original_step_intent=last_step.action.intent,
            tools=tool_info,
            believes=believes,
            planning_context=planning_context
        )

        result = json.loads(response)
        decomposed_steps = []
        
        for step_data in result.get("steps", []):
            decomposed_steps.append(StepDefinition(
                intent=step_data["intent"],
                tool=step_data["tool"],
                reason="decomposed",
                reason_details=f"Decomposed from: {last_step.action.intent}"
            ))

        return decomposed_steps

    async def _analyze_failure(self, failed_step: StepOutcome, executed_steps: list[StepOutcome],
                             goal: str, believes: list[str], tools: dict[str, ContextualTool],
                             planning_context: list[str]) -> dict:
        """
        Analyze a failed step with full execution history to determine recovery strategy.

        Args:
            failed_step: The step that failed
            executed_steps: All previously executed steps
            goal: The original goal
            believes: Current beliefs
            tools: Available tools
            planning_context: Relevant historical context for planning decisions

        Returns:
            Dictionary containing analysis, recommendations, recovery steps, and rerun decision
        """
        # Build tool info including input schemas for SchematicTools
        tool_info = {}
        for name, tool in tools.items():
            info: dict[str, str | dict | None] = {"description": tool.description}
            if isinstance(tool, SchematicToolAdapter):
                info["input_schema"] = tool.inner.input_schema
            tool_info[name] = info

        response = await self.call_template(
            "analyze_failure.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            tools=tool_info,
            planning_context=planning_context,
            executed_steps=[{
                "intent": outcome.action.intent,
                "tool": outcome.action.tool,
                "status": outcome.status.value,
                "results": outcome.results,
                "error_message": outcome.error_message,
                "discovered_insights": outcome.discovered_insights
            } for outcome in executed_steps],
            failed_step={
                "intent": failed_step.action.intent,
                "tool": failed_step.action.tool,
                "status": failed_step.status.value,
                "error_message": failed_step.error_message,
                "results": failed_step.results
            }
        )

        result = json.loads(response)

        # Log the failure analysis for debugging
        logger.info(f"Failure analysis: {result.get('analysis', 'No analysis provided')}")
        logger.info(f"Recovery recommendation: {result.get('recommendation', 'No recommendation provided')}")

        return result
