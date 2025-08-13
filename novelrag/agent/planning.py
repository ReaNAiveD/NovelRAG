import json
import logging

from .execution import ExecutableStep, StepDefinition, StepOutcome, StepStatus, ExecutionPlan
from .tool import ContextualTool, LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class PursuitPlanner(LLMToolMixin):
    """Responsible for planning and executing goals using the agent's tools."""
    
    def __init__(self, chat_llm: ChatLLM, template_env: TemplateEnvironment):
        super().__init__(chat_llm=chat_llm, template_env=template_env)

    async def create_initial_plan(self, goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Decompose a goal into executable steps based on available tools and beliefs.
        Args:
            goal: The goal to pursue.
            believes: Current beliefs of the agent.
            tools: Available tools for the agent.
        Returns:
            A list of ExecutableStep instances representing the decomposed steps.
        """
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

    async def adapt_plan(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Adapt the execution plan based on the last completed step and current context.
        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.
        Returns:
            A list of updated ExecutableStep instances representing the new schedule,
            with proper dependency relationships established between steps.
        """
        if last_step.status == StepStatus.SUCCESS:
            return await self._adapt_plan_from_success(last_step, original_plan, believes, tools)
        elif last_step.status == StepStatus.FAILED:
            return await self._adapt_plan_from_failure(last_step, original_plan, believes, tools)
        elif last_step.status == StepStatus.DECOMPOSED:
            return await self._adapt_plan_from_decomposition(last_step, original_plan, believes, tools)
        else:
            raise ValueError(f"Unexpected step status: {last_step.status.value}. Cannot adapt plan.")

    async def _adapt_plan_from_success(self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str], tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Adapt the execution plan based on the last completed step and current context.
        This method is specifically designed to handle cases where the last step was successful,
        allowing the agent to continue with the next steps in the plan.

        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.

        Returns:
            A list of updated ExecutableStep instances representing the new schedule,
            with proper dependency relationships established between steps.
        """
        executed_steps = original_plan.completed_steps + [last_step]
        planned_steps = [step.definition for step in last_step.triggered_actions]
        remaining_steps_from_scratch = await self._plan_remaining_steps(
            executed_steps=executed_steps,
            planned_steps=planned_steps,
            goal=original_plan.goal,
            believes=believes,
            tools=tools
        )
        # TODO: Determine the final steps
        raise NotImplementedError("This method needs to be implemented to handle success adaptation.")

    async def _adapt_plan_from_failure(self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str], tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Adapt the execution plan based on the last completed step and current context.
        This method is specifically designed to handle cases where the last step failed,
        allowing the agent to adjust its strategy and retry or modify steps as needed.

        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.

        Returns:
            A list of updated ExecutableStep instances representing the new schedule,
            with proper dependency relationships established between steps.
        """
        executed_steps = original_plan.completed_steps + [last_step]
        planned_steps = []
        remaining_steps_from_scratch = await self._plan_remaining_steps(
            executed_steps=executed_steps,
            planned_steps=planned_steps,
            goal=original_plan.goal,
            believes=believes,
            tools=tools
        )
        # TODO: Determine the final steps
        raise NotImplementedError("This method needs to be implemented to handle failure adaptation.")

    async def _adapt_plan_from_decomposition(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """ Adapt the execution plan based on the last completed step and current context,
        specifically when the last step was decomposed into multiple actions.

        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.
        Returns:
            A list of updated ExecutableStep instances representing the new schedule,
            with proper dependency relationships established between steps.
        """
        executed_steps = original_plan.completed_steps + [last_step]
        planned_steps = [step.definition for step in last_step.spawned_actions] + [last_step.action.definition]
        remaining_steps_from_scratch = await self._plan_remaining_steps(
            executed_steps=executed_steps,
            planned_steps=planned_steps,
            goal=original_plan.goal,
            believes=believes,
            tools=tools
        )
        # TODO: Determine the final steps
        raise NotImplementedError("This method needs to be implemented to handle decomposition adaptation.")


    async def _plan_remaining_steps(self, executed_steps: list[StepOutcome], planned_steps: list[StepDefinition],
                                    goal: str, believes: list[str], tools: dict[str, ContextualTool]) -> list[StepDefinition]:
        """
        Plan the remaining steps in the execution plan based on executed steps, planned steps and current context.
        This method analyzes the already executed steps and the planned steps to determine
        the next steps needed to achieve the goal, considering the agent's beliefs and available tools.

        Args:
            executed_steps: List of StepOutcome instances representing the steps that have already been executed.
            planned_steps: List of ExecutableStep instances representing the steps that are planned but not yet executed.
            goal: The goal to achieve with the execution plan.
            believes: The agent's current beliefs about the state of the world.
            tools: Dictionary mapping tool names to ContextualTool instances available for use.
        Returns:
            A list of updated ExecutableStep instances representing the new schedule,
            with proper dependency relationships established between steps.
        """
        key_steps = await self._find_key_steps(executed_steps, goal)
        key_results = [{"tool": step.action.tool, "intent": step.action.intent, "results": step.results, "errors": step.error_message} for step in key_steps]
        executed_steps = [{"tool": step.action.tool, "intent": step.action.intent, "status": step.status.value} for step in executed_steps]
        planned_steps = [{"tool": step.tool, "intent": step.intent} for step in planned_steps]
        response = await self.call_template(
            "plan_remaining_steps.jinja2",
            json_format=True,
            goal=goal,
            believes=believes,
            executed_steps=executed_steps,
            planned_steps=planned_steps,
            key_results=key_results,
            tools={name: tool.description for name, tool in tools.items()}
        )
        response = json.loads(response)
        new_steps = response["new_steps"]
        steps = [StepDefinition(**step) for step in new_steps]
        return steps

    async def adapt_plan_by_revision(
            self, last_step: StepOutcome, original_plan: ExecutionPlan, believes: list[str],
            tools: dict[str, ContextualTool]) -> list[ExecutableStep]:
        """
        Adapt the execution plan by revising steps based on the last completed step outcome.

        This method dynamically revises the execution plan by analyzing the outcome
        of the most recently completed step. It can delete obsolete pending steps and
        insert new steps to adapt to changing circumstances. The method uses the
        "reschedule_steps.jinja2" template to determine which existing steps to remove
        (via delete_count) and what new steps to add.

        The revision process:
        - Analyzes the last step's outcome (success/failure/decomposition)
        - Determines which pending steps are no longer relevant (delete_count)
        - Generates new steps based on current context and beliefs
        - Handles step reruns for failed steps
        - Maintains proper dependency relationships between all steps

        Args:
            last_step: The outcome of the most recently completed step, including
                      status, results, spawned actions, and any error information.
            original_plan: The original execution plan that was being followed before the last step.
            believes: The agent's current beliefs about the state of the world,
                     which influence step creation and modification decisions.
            tools: Dictionary mapping tool names to ContextualTool instances
                  that are available for use in new or modified steps.

        Returns:
            A list of updated ExecutableStep instances representing the revised plan,
            combining spawned steps, rerun steps (if needed), new steps, and remaining target steps.

        Note:
            This method differs from the status-based adapt_plan method by using a unified
            revision approach that can handle any step outcome type through template logic.
        """
        prev_steps = original_plan.completed_steps
        target_steps = original_plan.pending_steps[1:]
        goal = original_plan.goal
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
        if last_step.status != StepStatus.SUCCESS:
            # If the last step was not successful, we need to ensure it is included in the schedule
            rerun_step = [ExecutableStep(
                definition=StepDefinition(
                    tool=last_step.action.tool,
                    intent=last_step.action.intent,
                    step_id=last_step.action.step_id,
                    progress=last_step.progress
                ),
                contribute_to=last_step.action.contribute_to,
                spawned_by=last_step.action.spawned_by,
                triggered_by=last_step.action.triggered_by,
            )]
        else:
            rerun_step = []
        return spawned_steps + (rerun_step or triggered_steps) + steps + target_steps

    async def _find_key_steps(self, executed_steps: list[StepOutcome], goal: str, threshold: int = 3) -> list[StepOutcome]:
        """Identify key steps in the execution plan based on their contribution to the decision of the next step."""
        if not executed_steps:
            return []
        steps = [{"tool": step.action.tool, "intent": step.action.intent, "status": step.status.value} for step in executed_steps[:-1]]
        last_step = executed_steps[-1]
        if last_step.status == StepStatus.SUCCESS:
            steps.append({
                "tool": last_step.action.tool,
                "intent": last_step.action.intent,
                "status": last_step.status.value,
                "results": last_step.results,
                "triggered_actions": [step.intent for step in last_step.triggered_actions]
            })
        elif last_step.status == StepStatus.FAILED:
            steps.append({
                "tool": last_step.action.tool,
                "intent": last_step.action.intent,
                "status": last_step.status.value,
                "progress": last_step.progress,
                "error_message": last_step.error_message
            })
        elif last_step.status == StepStatus.DECOMPOSED:
            steps.append({
                "tool": last_step.action.tool,
                "intent": last_step.action.intent,
                "status": last_step.status.value,
                "spawned_actions": [step.intent for step in last_step.spawned_actions]
            })
        response = await self.call_template(
            "identify_key_steps_for_planning.jinja2",
            json_format=True,
            goal=goal,
            steps=steps,
            threshold=threshold
        )
        key_step_indices = json.loads(response)['indices']
        key_steps = [executed_steps[i] for i in key_step_indices if i < len(executed_steps)]
        return key_steps

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
