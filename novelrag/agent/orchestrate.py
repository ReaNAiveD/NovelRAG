import json
from dataclasses import dataclass, field
from novelrag.agent.steps import StepOutcome
from novelrag.agent.tool import LLMToolMixin, SchematicTool
from novelrag.agent.workspace import ResourceContext
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment


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
    
    # For execute decision
    execution: dict | None = None  # {"tool": str, "params": dict, "confidence": str}
    
    # For finalize decision  
    finalization: dict | None = None  # {"status": str, "response": str, "evidence": list, "gaps": list}
    
    # Analysis details for refinement phase
    context_verification: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementDecision:
    """Result from Phase 2: Refinement Analysis."""
    analysis: dict  # Quality assessment and discovered issues
    verdict: str  # "approve" or "refine"
    
    # For approval
    approval: dict | None = None  # {"ready": bool, "confidence": str, "notes": str}
    
    # For refinement
    refinement: dict | None = None  # {"refined_goal": str, "additions": list, "exploration_hints": dict, "rationale": str}


@dataclass(frozen=True)
class OrchestrationExecutionPlan:
    reason: str
    tool: str
    params: dict = field(default_factory=dict)
    future_steps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OrchestrationFinalization:
    reason: str
    response: str
    status: str  # success, failed, abandoned


class OrchestrationLoop(LLMToolMixin):
    def __init__(self, context: ResourceContext, template_env: TemplateEnvironment, chat_llm: ChatLLM, max_iter: int | None = 5, min_iter: int | None = None):
        super().__init__(template_env, chat_llm)
        self.max_iter = max_iter
        self.min_iter: int = min_iter or 0
        self.context = context
        self.expanded_tools: set[str] = set()

    async def _discover_and_expand_context(
            self,
            user_request: str,
            goal: str,
            available_tools: dict[str, SchematicTool],
    ) -> DiscoveryPlan:
        segments, nonexisted = await self._get_workspace_segments()
        expanded_tool_info = self._get_expanded_tools(available_tools)
        collapsed_tool_names = list(set(available_tools.keys()) - self.expanded_tools)
        
        discovery_json = await self.call_template(
            "context_discovery.jinja2",
            json_format=True,
            user_request=user_request,
            goal=goal,
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
            goal: str,
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
            user_request: str,
            goal: str,
            completed_steps: list[StepOutcome],
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
            user_request=user_request,
            goal=goal,
            completed_steps=[{
                "tool": step.action.tool,
                "intent": step.action.reason,
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
            user_request: str,
            original_goal: str,
            action_decision: ActionDecision,
            completed_steps: list[StepOutcome],
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
            user_request=user_request,
            original_goal=original_goal,
            action_decision={
                "situation_analysis": action_decision.situation_analysis,
                "decision": action_decision.decision_type,
                "execution": action_decision.execution,
                "finalization": action_decision.finalization,
                "context_verification": action_decision.context_verification
            },
            completed_steps=[{
                "tool": step.action.tool,
                "intent": step.action.reason,
                "status": step.status.value,
                "results": step.results
            } for step in completed_steps],
            workspace_segments=segments,
            available_tools=expanded_tool_info,
            collapsed_tools=collapsed_tool_names
        )
        result = json.loads(refinement_json)
        
        return RefinementDecision(
            analysis=result.get("analysis", {}),
            verdict=result.get("verdict", "approve"),
            approval=result.get("approval"),
            refinement=result.get("refinement")
        )
    
    async def _apply_context_gap_suggestions(self, suggestions: dict):
        """Apply suggested actions from refinement exploration hints."""
        for query in suggestions.get("search_terms", []):
            await self.context.search_resources(query)
        
        for uri in suggestions.get("resource_paths", []):
            await self.context.query_resource(uri)
        
        # Add support for tool expansion suggestions
        for tool_name in suggestions.get("tools_to_expand", []):
            self.expanded_tools.add(tool_name)
    
    def _convert_to_orchestration_action(
            self, 
            action_decision: ActionDecision
    ) -> OrchestrationExecutionPlan | OrchestrationFinalization:
        """Convert ActionDecision to orchestration type."""
        if action_decision.decision_type == "execute" and action_decision.execution:
            exec_data = action_decision.execution
            return OrchestrationExecutionPlan(
                reason=action_decision.situation_analysis,
                tool=exec_data["tool"],
                params=exec_data["params"],
                future_steps=[]
            )
        elif action_decision.decision_type == "finalize" and action_decision.finalization:
            final_data = action_decision.finalization
            return OrchestrationFinalization(
                reason=action_decision.situation_analysis,
                response=final_data.get("response", ""),
                status=final_data.get("status", "success")
            )
        else:
            # Fallback for malformed decision
            return OrchestrationFinalization(
                reason="Invalid action decision",
                response="Unable to process the action decision.",
                status="failed"
            )

    async def execution_advance(
            self,
            user_request: str,
            goal: str,
            completed_steps: list[StepOutcome],
            pending_steps: list[str],
            available_tools: dict[str, SchematicTool],
    ) -> OrchestrationExecutionPlan | OrchestrationFinalization:
        """
        Advance execution through phased context refinement and planning.
        
        Uses a two-phase decision architecture:
        1. Action Decision - Decides to execute tool or finalize
        2. Refinement Analysis - Approves action or refines goal for next iteration
        
        The goal evolves through iterations to incorporate discovered requirements.
        """        
        iter_num = 0
        current_goal = goal
        last_planned_action: OrchestrationExecutionPlan | OrchestrationFinalization = OrchestrationFinalization(
            reason="Maximum iterations reached without achieving goal",
            response="I was unable to complete your request within the iteration limit.",
            status="abandoned"
        )
        
        while True:
            # Context discovery and refinement loop
            while True:
                iter_num += 1
                discovery_plan = await self._discover_and_expand_context(
                    user_request, current_goal, available_tools
                )
                await self._apply_discovery_plan(discovery_plan)

                if not discovery_plan.refinement_needed:
                    break
                if self.max_iter is not None and iter_num >= self.max_iter:
                    break
                
                refinement_plan = await self._filter_and_refine_context(
                    current_goal, available_tools, discovery_plan.discovery_analysis
                )
                await self._apply_refinement_plan(refinement_plan)
            
            action_decision = await self._make_action_decision(
                user_request, current_goal, completed_steps, available_tools
            )

            refinement_decision = await self._analyze_and_refine(
                user_request, current_goal, action_decision, completed_steps, available_tools
            )
            
            # Convert action decision to orchestration type
            planned_action = self._convert_to_orchestration_action(action_decision)
            last_planned_action = planned_action
            
            # Process refinement verdict
            if refinement_decision.verdict == "approve":
                # Return approved action if min iterations met
                if isinstance(planned_action, OrchestrationExecutionPlan) and iter_num >= self.min_iter:
                    return planned_action
                elif isinstance(planned_action, OrchestrationFinalization):
                    return planned_action
            else:
                # Refine goal and continue iteration
                if refinement_decision.refinement:
                    refinement = refinement_decision.refinement
                    current_goal = refinement.get("refined_goal", current_goal)
                    
                    # Apply exploration hints
                    if exploration := refinement.get("exploration_hints"):
                        await self._apply_context_gap_suggestions(exploration)
            
            # Check iteration limits
            if self.max_iter is not None and iter_num >= self.max_iter:
                break
        
        # Fallback: Return last planned action (already properly formatted)
        return last_planned_action
    
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
