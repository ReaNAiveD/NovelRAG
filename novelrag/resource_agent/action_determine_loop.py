"""Orchestration loop for resource-aware action determination.

This module provides the OrchestrationLoop which implements ActionDeterminer
using multi-phase context discovery, refinement, and decision-making.
"""

import json
from dataclasses import dataclass, field

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import ActionDeterminer, PursuitAssessment, PursuitAssessor, PursuitProgress
from novelrag.agenturn.step import OperationPlan, OperationOutcome, Resolution
from novelrag.agenturn.tool import SchematicTool
from novelrag.llm import LLMMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .workspace import ResourceContext


@dataclass(frozen=True)
class DiscoveryPlan:
    """Result from context discovery phase."""
    discovery_analysis: str = field(default="")
    search_queries: list[str] = field(default_factory=list)
    query_resources: list[str] = field(default_factory=list)
    expand_tools: list[str] = field(default_factory=list)

    @property
    def refinement_needed(self) -> bool:
        return bool(
            self.search_queries or
            self.query_resources
        )


@dataclass(frozen=True)
class RefinementPlan:
    """Result from context refinement phase."""
    exclude_resources: list[str] = field(default_factory=list)
    exclude_properties: list[dict] = field(default_factory=list)  # [{"uri": str, "property": str}]
    collapse_tools: list[str] = field(default_factory=list)
    sorted_segments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActionDecision:
    """Result from Phase 1: Action Decision."""
    situation_analysis: str
    decision_type: str  # "execute" or "finalize"
    execution: dict | None = None  # {"tool": str, "params": dict, "confidence": str}
    finalization: dict | None = None  # {"status": str, "response": str, "evidence": list, "gaps": list}
    context_verification: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementDecision:
    """Result from Phase 2: Refinement Analysis."""
    analysis: dict  # Quality assessment and discovered issues
    verdict: str  # "approve" or "refine"
    approval: dict | None = None  # {"ready": bool, "confidence": str, "notes": str}
    refinement: dict | None = None  # Refer to Pursuit Assessment structure


class ActionDetermineLoop(ActionDeterminer, LLMMixin):
    """Resource-aware action determiner using multi-phase orchestration.
    
    Implements the ActionDeterminer protocol for use with GoalExecutor.
    Uses a four-phase decision architecture:
    1. Context Discovery - Find relevant context
    2. Context Refinement - Filter and prioritize context
    3. Action Decision - Decide to execute tool or finalize
    4. Refinement Analysis - Validate decision or refine goal
    """
    
    def __init__(self, context: ResourceContext, chat_llm: ChatLLM, template_lang: str = 'en', max_iter: int | None = 5, min_iter: int | None = None):
        template_env = TemplateEnvironment(package_name="novelrag.resource_agent", default_lang=template_lang,)
        LLMMixin.__init__(self, template_env=template_env, chat_llm=chat_llm)
        self.assessor = PursuitAssessor(chat_llm=chat_llm, lang=template_lang)
        self.max_iter = max_iter
        self.min_iter: int = min_iter or 0
        self.context = context
        self.expanded_tools: set[str] = set()

    async def determine_action(
            self,
            beliefs: list[str],
            pursuit_progress: PursuitProgress,
            available_tools: dict[str, SchematicTool]
    ) -> OperationPlan | Resolution:
        """
        Advance execution through phased context refinement and planning.

        Uses a two-phase decision architecture:
        1. Action Decision - Decides to execute tool or finalize
        2. Refinement Analysis - Approves action or refines goal for next iteration

        The goal evolves through iterations to incorporate discovered requirements.
        """
        iter_num = 0
        goal = pursuit_progress.goal
        pursuit_assessment = await self.assessor.assess_progress(
            pursuit=pursuit_progress,
            beliefs=beliefs,
        )
        last_planned_action: OperationPlan | Resolution = Resolution(
            reason="Maximum iterations reached without achieving goal",
            response="I was unable to complete your request within the iteration limit.",
            status="abandoned"
        )

        while True:
            # Context discovery and refinement loop
            while True:
                iter_num += 1
                discovery_plan = await self._discover_and_expand_context(
                    goal, pursuit_assessment, available_tools
                )
                await self._apply_discovery_plan(discovery_plan)

                if not discovery_plan.refinement_needed:
                    break
                if self.max_iter is not None and iter_num >= self.max_iter:
                    break
                
                refinement_plan = await self._filter_and_refine_context(
                    goal, pursuit_assessment, available_tools, discovery_plan.discovery_analysis
                )
                await self._apply_refinement_plan(refinement_plan)

            action_decision = await self._make_action_decision(
                goal, pursuit_assessment, pursuit_progress.executed_steps, available_tools
            )
            planned_action = self._convert_to_orchestration_action(action_decision)
            last_planned_action = planned_action
            refinement_decision = await self._analyze_and_refine(
                goal, pursuit_assessment, action_decision, pursuit_progress.executed_steps, available_tools
            )
            
            # Process refinement verdict
            if refinement_decision.verdict == "approve":
                # Return approved action if min iterations met
                if isinstance(planned_action, OperationPlan) and iter_num >= self.min_iter:
                    return planned_action
                elif isinstance(planned_action, Resolution):
                    return planned_action
            else:
                # Refine goal and continue iteration
                if refinement_decision.refinement:
                    refinement = refinement_decision.refinement
                    pursuit_assessment = PursuitAssessment(
                        finished_tasks=refinement.get("finished_tasks", pursuit_assessment.finished_tasks),
                        remaining_work_summary=refinement.get("remaining_work_summary", pursuit_assessment.remaining_work_summary),
                        required_context=refinement.get("required_context", pursuit_assessment.required_context),
                        expected_actions=refinement.get("expected_actions", pursuit_assessment.expected_actions),
                        boundary_conditions=refinement.get("boundary_conditions", pursuit_assessment.boundary_conditions),
                        exception_conditions=refinement.get("exception_conditions", pursuit_assessment.exception_conditions),
                        success_criteria=refinement.get("success_criteria", pursuit_assessment.success_criteria),
                    )
            
            # Check iteration limits
            if self.max_iter is not None and iter_num >= self.max_iter:
                break
        
        # Fallback: Return last planned action (already properly formatted)
        return last_planned_action

    async def _discover_and_expand_context(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            available_tools: dict[str, SchematicTool],
    ) -> DiscoveryPlan:
        segments, nonexisted = await self._get_workspace_segments()
        expanded_tool_info = self._get_expanded_tools(available_tools)
        collapsed_tool_names = list(set(available_tools.keys()) - self.expanded_tools)
        
        discovery_json = await self.call_template(
            "context_discovery.jinja2",
            json_format=True,
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            workspace_segments=segments,
            search_history=self.context.search_history[-10:],  # Last 10 searches
            nonexisted_uris=nonexisted,
            expanded_tools=expanded_tool_info,
            collapsed_tools=collapsed_tool_names
        )
        discovery_result = json.loads(discovery_json)
        
        return DiscoveryPlan(
            discovery_analysis=discovery_result.get("discovery_analysis", ""),
            search_queries=discovery_result.get("search_queries", []),
            query_resources=discovery_result.get("query_resources", []),
            expand_tools=discovery_result.get("expand_tools", [])
        )
    
    async def _apply_discovery_plan(self, plan: DiscoveryPlan):
        for query in plan.search_queries:
            await self.context.search_resources(query)
        
        for uri in plan.query_resources:
            await self.context.query_resource(uri)
        
        self.expanded_tools.update(plan.expand_tools)
    
    async def _filter_and_refine_context(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            available_tools: dict[str, SchematicTool],
            discovery_analysis: str = "",
    ) -> RefinementPlan:
        segments, _ = await self._get_workspace_segments()
        expanded_tool_info = self._get_expanded_tools(available_tools)
        collapsed_tool_names = list(set(available_tools.keys()) - self.expanded_tools)
        
        relevance_json = await self.call_template(
            "context_relevance.jinja2",
            json_format=True,
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            discovery_analysis=discovery_analysis,
            workspace_segments=segments,
            expanded_tools=expanded_tool_info,
            collapsed_tools=collapsed_tool_names
        )
        relevance_result = json.loads(relevance_json)
        
        return RefinementPlan(
            exclude_resources=relevance_result.get("exclude_resources", []),
            exclude_properties=relevance_result.get("exclude_properties", []),
            collapse_tools=relevance_result.get("collapse_tools", []),
            sorted_segments=relevance_result.get("sorted_segments", [])
        )
    
    async def _apply_refinement_plan(self, plan: RefinementPlan):
        for uri in plan.exclude_resources:
            await self.context.exclude_resource(uri)
        
        for item in plan.exclude_properties:
            await self.context.exclude_property(item["uri"], item["property"])
        
        self.expanded_tools.difference_update(plan.collapse_tools)
        
        if plan.sorted_segments:
            await self.context.sort_resources(plan.sorted_segments)
    
    async def _make_action_decision(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            completed_steps: list[OperationOutcome],
            available_tools: dict[str, SchematicTool],
    ) -> ActionDecision:
        """
        Determine whether to execute a tool or finalize.
        Always returns a decisive action (execute or finalize).
        """
        segments, _ = await self._get_workspace_segments()
        expanded_tool_info = self._get_expanded_tools(available_tools)
        
        decision_json = await self.call_template(
            "action_decision.jinja2",
            json_format=True,
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            completed_steps=[{
                "tool": step.operation.tool,
                "intent": step.operation.reason,
                "status": step.status.value,
                "results": step.results
            } for step in completed_steps],
            workspace_segments=segments,
            available_tools=expanded_tool_info
        )
        result = json.loads(decision_json)
        
        return ActionDecision(
            situation_analysis=result.get("situation_analysis", ""),
            decision_type=result.get("decision", "finalize"),
            execution=result.get("execution"),
            finalization=result.get("finalization"),
            context_verification=result.get("context_verification", {})
        )

    async def _analyze_and_refine(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            action_decision: ActionDecision,
            completed_steps: list[OperationOutcome],
            available_tools: dict[str, SchematicTool],
    ) -> RefinementDecision:
        """
        Analyze the action decision and determine if refinement is needed.
        Either approves execution or generates refined goal for next iteration.
        """
        segments, _ = await self._get_workspace_segments()
        expanded_tool_info = self._get_expanded_tools(available_tools)
        collapsed_tool_names = list(set(available_tools.keys()) - self.expanded_tools)
        
        refinement_json = await self.call_template(
            "refinement_analysis.jinja2",
            json_format=True,
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            action_decision={
                "situation_analysis": action_decision.situation_analysis,
                "decision": action_decision.decision_type,
                "execution": action_decision.execution,
                "finalization": action_decision.finalization,
                "context_verification": action_decision.context_verification
            },
            completed_steps=[{
                "tool": step.operation.tool,
                "intent": step.operation.reason,
                "status": step.status.value,
                "results": step.results
            } for step in completed_steps],
            workspace_segments=segments,
            available_tools=expanded_tool_info,
            collapsed_tools=collapsed_tool_names
        )
        result: dict = json.loads(refinement_json)
        
        return RefinementDecision(
            analysis=result.get("analysis", {}),
            verdict=result.get("verdict", "approve"),
            approval=result.get("approval"),
            refinement=result.get("refinement")
        )

    def _convert_to_orchestration_action(
            self, 
            action_decision: ActionDecision
    ) -> OperationPlan | Resolution:
        """Convert ActionDecision to orchestration type."""
        if action_decision.decision_type == "execute" and action_decision.execution:
            exec_data = action_decision.execution
            return OperationPlan(
                reason=action_decision.situation_analysis,
                tool=exec_data["tool"],
                parameters=exec_data["params"]
            )
        elif action_decision.decision_type == "finalize" and action_decision.finalization:
            final_data = action_decision.finalization
            return Resolution(
                reason=action_decision.situation_analysis,
                response=final_data.get("response", ""),
                status=final_data.get("status", "success")
            )
        else:
            # Fallback for malformed decision
            return Resolution(
                reason="Invalid action decision",
                response="Unable to process the action decision.",
                status="failed"
            )

    async def _get_workspace_segments(self):
        """Prepare workspace segments with enriched property information."""
        segments = []
        nonexisted = []
        
        for segment in self.context.workspace.sorted_segments():
            data = await self.context.build_segment_data(segment)
            if data:
                # Enrich segment data with clearer structure
                enriched = {
                    "uri": data["uri"],
                    "included_data": data.get("included_data", {}),
                    "pending_properties": data.get("pending_properties", []),
                    "child_ids": data.get("child_ids", {}),
                    "relations": data.get("relations", {}),
                    # Add computed fields for easier template consumption
                    "total_children": sum(len(ids) for ids in data.get("child_ids", {}).values()),
                }
                segments.append(enriched)
            else:
                nonexisted.append(segment.uri)
        
        return segments, nonexisted

    def _get_expanded_tools(self, available_tools: dict[str, SchematicTool]):
        """Prepare only expanded tools with full schema details."""
        tools = {}
        for name in self.expanded_tools:
            if name not in available_tools:
                continue
                
            tool = available_tools[name]
            tools[name] = {
                "description": tool.description,
                "prerequisites": tool.prerequisites,
                "output_description": tool.output_description,
                "input_schema": tool.input_schema,
                "required_params": tool.input_schema.get("required", []) if tool.input_schema else []
            }
            
            # Add schema_params for easier template consumption
            if tool.input_schema and "properties" in tool.input_schema:
                tools[name]["schema_params"] = {
                    param: {
                        "description": details.get("description", ""),
                        "type": details.get("type", ""),
                        "required": param in tool.input_schema.get("required", [])
                    }
                    for param, details in tool.input_schema["properties"].items()
                }
        return tools
    
    def _format_response(self, response_framework: dict):
        """Format final response from framework."""
        parts = [response_framework.get("main_answer", "")]
        
        if points := response_framework.get("key_points"):
            parts.append("\n\nKey Points:")
            parts.extend(f"- {point}" for point in points)
        
        if context_refs := response_framework.get("supporting_context"):
            parts.append("\n\nSupporting Context:")
            parts.extend(f"- {ref}" for ref in context_refs)
        
        return "\n".join(filter(None, parts))
