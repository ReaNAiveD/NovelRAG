import json
import logging

from .execution import ExecutableStep, StepDefinition, StepOutcome, StepStatus
from .tool import ContextualTool, LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class GoalPlanner(LLMToolMixin):
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

    async def reschedule_plan(
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


