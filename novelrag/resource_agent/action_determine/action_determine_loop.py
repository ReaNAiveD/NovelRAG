"""Orchestration loop for resource-aware action determination.

This module provides the OrchestrationLoop which implements ActionDeterminer
using multi-phase context discovery, refinement, and decision-making.
"""

import logging
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import ActionDeterminer, PursuitAssessment, PursuitAssessor, PursuitProgress
from novelrag.agenturn.step import OperationPlan, OperationOutcome, Resolution
from novelrag.agenturn.tool import SchematicTool
from novelrag.agenturn.types import InteractionContext
from novelrag.resource_agent.workspace import ResourceContext, ContextSnapshot, SegmentData, SearchHistoryItem

logger = logging.getLogger(__name__)


class DiscoveryPlan(BaseModel):
    """Plan for discovering and expanding context relevant to achieving the current goal.

    Produced by the context discovery phase, this plan specifies what resources
    to load, what searches to perform, and which tools to expand for the next iteration.
    """
    discovery_analysis: Annotated[str, Field(
        description=(
            "Analysis of current context coverage: what has been found so far, "
            "what unloaded children/relations remain to explore, what new searches "
            "are needed, and which tools are relevant given the goal and expected actions."
        ),
    )]
    search_queries: Annotated[list[str], Field(
        description=(
            "New search terms for finding resources whose URIs are unknown. "
            "Must not repeat or closely resemble previous searches. "
            "Consider semantic variants and translations (e.g., protagonist → 主角). "
            "Use an empty array when all relevant concepts have been searched."
        ),
    )]
    query_resources: Annotated[list[str], Field(
        description=(
            "Resource URIs to load from the workspace, constructed only from visible "
            "Children or Relations in the current knowledge base. "
            "For children: parent_uri + '/' + child_id (e.g., '/cne/character_template'). "
            "For relations: use the full URI as shown. "
            "Never invent or guess URIs that are not listed."
        ),
    )]
    expand_tools: Annotated[list[str], Field(
        description=(
            "Names of currently collapsed tools whose full schemas should be made visible. "
            "Expand tools proactively when the goal or expected actions imply "
            "resource creation, modification, or linking — they cannot be used unless expanded."
        ),
    )]

    @property
    def refinement_needed(self) -> bool:
        return bool(
            self.search_queries or
            self.query_resources
        )


class ResourceProperty(BaseModel):
    """A specific property to exclude from a loaded resource."""

    uri: Annotated[str, Field(
        description="URI of the loaded resource (must appear in the Loaded Resources list).",
    )]
    property: Annotated[str, Field(
        description="Name of the visible property to hide from this resource.",
    )]


class RefinementPlan(BaseModel):
    """Plan for filtering and prioritizing loaded context to maximize signal-to-noise.

    Produced by the context relevance phase after discovery. Specifies which
    resources/properties to exclude, which tool schemas to collapse, and the
    relevance-ordered presentation of remaining resources.
    """
    relevance_analysis: Annotated[str, Field(
        description=(
            "Per-resource relevance rating (CRITICAL/HIGH/MEDIUM/LOW/NONE) with brief justification. "
            "Consider semantic relationships, translations, and the requirements of expanded tools."
        ),
    )]
    exclude_resources: Annotated[list[str], Field(
        description=(
            "URIs of loaded resources rated NONE relevance — completely unrelated to the current goal. "
            "Be conservative: only exclude clearly irrelevant items."
        ),
    )]
    exclude_properties: Annotated[list[ResourceProperty], Field(
        description=(
            "Specific visible properties to hide from kept resources. "
            "Target LOW-relevance properties that add noise without aiding the goal or tool inputs."
        ),
    )]
    collapse_tools: Annotated[list[str], Field(
        description=(
            "Names of currently expanded tools whose schemas should be hidden. "
            "Only collapse tools genuinely unrelated to the goal — do NOT collapse tools "
            "needed for expected actions."
        ),
    )]
    sorted_segments: Annotated[list[str], Field(
        description=(
            "All loaded resource URIs ordered by relevance (most relevant first). "
            "Must include every URI from the Loaded Resources list — this is a complete reordering, not a subset."
        ),
    )]


class ExecutionDetail(BaseModel):
    """Details for tool execution."""
    tool: Annotated[str, Field(description="Name of the tool to execute.")]
    params: Annotated[dict, Field(description="Parameters to pass to the tool, using exact values from context.")] = {}
    confidence: Annotated[str, Field(description="Confidence level: high, medium, or low.")] = "medium"
    reasoning: Annotated[str, Field(description="Why this execution will achieve the goal with current context.")] = ""


class FinalizationDetail(BaseModel):
    """Details for finalizing with a response."""
    status: Annotated[str, Field(description="Outcome status: success, failed, or incomplete.")]
    response: Annotated[str, Field(description="Complete user-facing response explaining the outcome.")]
    evidence: Annotated[list[str], Field(description="References to specific segments with supporting information.")] = []
    gaps: Annotated[list[str], Field(description="Specific missing information (only if status is incomplete).")] = []


class ActionDecision(BaseModel):
    """Action decision: execute a tool or finalize with a response."""
    situation_analysis: Annotated[str, Field(
        description="Comprehensive assessment: what we have, what we need, what we can do.",
    )]
    decision_type: Annotated[str, Field(
        description="The action to take: 'execute' to run a tool, or 'finalize' to provide a response.",
    )]
    execution: Annotated[ExecutionDetail | None, Field(
        description="Tool execution details. Required when decision_type is 'execute'.",
    )] = None
    finalization: Annotated[FinalizationDetail | None, Field(
        description="Finalization details. Required when decision_type is 'finalize'.",
    )] = None
    context_verification: Annotated[dict, Field(
        description="Verification of prerequisites and parameter mapping against context segments.",
    )] = {}


class RefinementApproval(BaseModel):
    """Approval details when the action decision is approved."""
    ready: Annotated[bool, Field(description="Whether the action is ready to execute.")]
    confidence: Annotated[str, Field(description="Confidence level: high or medium.")]
    notes: Annotated[str, Field(description="Any caveats or considerations.")] = ""


class RefinementDecision(BaseModel):
    """Refinement analysis: approve the action or refine the approach."""
    analysis: Annotated[str, Field(
        description=(
            "Quality assessment including decision_quality, prerequisite_verification, "
            "parameter_verification, discovered_issues, and alternative_approaches."
        ),
    )]
    verdict: Annotated[str, Field(
        description="The verdict: 'approve' to proceed with the action, or 'refine' to adjust the approach.",
    )]
    approval: Annotated[RefinementApproval | None, Field(
        description="Approval details. Required when verdict is 'approve'.",
    )] = None
    refinement: Annotated[PursuitAssessment | None, Field(
        description="Refined pursuit assessment. Required when verdict is 'refine'.",
    )] = None


class ContextDiscoverer(Protocol):
    """Protocol for context discovery phase."""
    async def discover(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            workspace_segment: list[SegmentData],
            non_existed_uris: list[str],
            search_history: list[SearchHistoryItem],
            expanded_tools: dict[str, SchematicTool],
            collapsed_tools: dict[str, SchematicTool],
    ) -> DiscoveryPlan:
        ...


class ContextAnalyser(Protocol):
    """Protocol for context refinement phase."""
    async def analyse(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            workspace_segment: list[SegmentData],
            expanded_tools: dict[str, SchematicTool],
            collapsed_tools: dict[str, SchematicTool],
            discovery_analysis: str
    ) -> RefinementPlan:
        ...


class ActionDecider(Protocol):
    """Protocol for action decision phase."""
    async def decide(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            completed_steps: list[OperationOutcome],
            workspace_segment: list[SegmentData],
            expanded_tools: dict[str, SchematicTool],
    ) -> ActionDecision:
        ...


class RefinementAnalyzer(Protocol):
    """Protocol for refinement analysis phase."""
    async def analyze(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            action_decision: ActionDecision,
            completed_steps: list[OperationOutcome],
            workspace_segment: list[SegmentData],
            expanded_tools: dict[str, SchematicTool],
            collapsed_tools: dict[str, SchematicTool],
    ) -> RefinementDecision:
        ...


class ActionDetermineLoop(ActionDeterminer):
    """Resource-aware action determiner using multi-phase orchestration.
    
    Implements the ActionDeterminer protocol for use with GoalExecutor.
    Uses a four-phase decision architecture:
    1. Context Discovery - Find relevant context
    2. Context Refinement - Filter and prioritize context
    3. Action Decision - Decide to execute tool or finalize
    4. Refinement Analysis - Validate decision or refine goal
    """
    
    def __init__(
            self,
            context: ResourceContext,
            pursuit_assessor: PursuitAssessor,
            discoverer: ContextDiscoverer,
            analyser: ContextAnalyser,
            decider: ActionDecider,
            refiner: RefinementAnalyzer,
            max_iter: int | None = 5,
            min_iter: int | None = None,
    ):
        self.max_iter = max_iter
        self.min_iter: int = min_iter or 0
        self.context = context
        self.expanded_tools: set[str] = set()
        self.assessor = pursuit_assessor
        self.discoverer = discoverer
        self.analyser = analyser
        self.decider = decider
        self.refiner = refiner

    async def determine_action(
            self,
            beliefs: list[str],
            pursuit_progress: PursuitProgress,
            available_tools: dict[str, SchematicTool],
            interaction_history: InteractionContext | None = None,
    ) -> OperationPlan | Resolution:
        """
        Advance execution through phased context refinement and planning.

        Uses a two-phase decision architecture:
        1. Action Decision - Decides to execute tool or finalize
        2. Refinement Analysis - Approves action or refines goal for next iteration

        The goal evolves through iterations to incorporate discovered requirements.
        """
        goal = pursuit_progress.goal
        pursuit_assessment = await self.assessor.assess_progress(
            pursuit=pursuit_progress,
            beliefs=beliefs,
            interaction_history=interaction_history,
        )
        return await self._action_loop(
            goal, pursuit_assessment,
            pursuit_progress.executed_steps, available_tools,
        )

    async def _action_loop(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            completed_steps: list[OperationOutcome],
            available_tools: dict[str, SchematicTool],
    ) -> OperationPlan | Resolution:
        """Main action decision loop with refinement."""
        iter_num = 0
        last_planned_action: OperationPlan | Resolution = Resolution(
            reason="Maximum iterations reached without achieving goal",
            response="I was unable to complete your request within the iteration limit.",
            status="abandoned"
        )

        while True:
            iter_num = await self._context_discovery_loop(
                goal, pursuit_assessment, available_tools, iter_num,
            )
            logger.info(f"Context discovery completed for iteration {iter_num}.")

            # Make action decision
            ctx = await self.context.snapshot()
            expanded_tools = self._get_tool_map(available_tools, expanded=True)
            action_decision = await self.decider.decide(
                goal=goal,
                pursuit_assessment=pursuit_assessment,
                completed_steps=completed_steps,
                workspace_segment=ctx.segments,
                expanded_tools=expanded_tools,
            )
            planned_action = self._convert_to_orchestration_action(action_decision)
            logger.info(f"Action decision made: {planned_action} on iteration {iter_num}.")
            last_planned_action = planned_action

            # Analyze and refine decision
            ctx = await self.context.snapshot()
            expanded_tools = self._get_tool_map(available_tools, expanded=True)
            collapsed_tools = self._get_tool_map(available_tools, expanded=False)
            refinement_decision = await self.refiner.analyze(
                goal=goal,
                pursuit_assessment=pursuit_assessment,
                action_decision=action_decision,
                completed_steps=completed_steps,
                workspace_segment=ctx.segments,
                expanded_tools=expanded_tools,
                collapsed_tools=collapsed_tools,
            )

            # Process refinement verdict
            if refinement_decision.verdict == "approve":
                logger.info(f"Action decision approved.")
                if isinstance(planned_action, OperationPlan) and iter_num >= self.min_iter:
                    return planned_action
                elif isinstance(planned_action, Resolution):
                    return planned_action
            else:
                logger.info(f"Action decision requires refinement: {refinement_decision.analysis}")
                if refinement_decision.refinement:
                    pursuit_assessment = refinement_decision.refinement

            if self.max_iter is not None and iter_num >= self.max_iter:
                break

        return last_planned_action

    async def _context_discovery_loop(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            available_tools: dict[str, SchematicTool],
            iter_num: int,
    ) -> int:
        """Discover and refine context through iterative search and filtering.

        Returns the updated iteration count.
        """
        while True:
            iter_num += 1

            # Discover context
            ctx = await self.context.snapshot()
            expanded_tools = self._get_tool_map(available_tools, expanded=True)
            collapsed_tools = self._get_tool_map(available_tools, expanded=False)
            discovery_plan = await self.discoverer.discover(
                goal=goal,
                pursuit_assessment=pursuit_assessment,
                workspace_segment=ctx.segments,
                non_existed_uris=ctx.nonexistent_uris,
                search_history=self.context.search_history[-10:],
                expanded_tools=expanded_tools,
                collapsed_tools=collapsed_tools,
            )
            await self._apply_discovery_plan(discovery_plan)
            if discovery_plan.refinement_needed:
                logger.info(f"Identified need for resource: {discovery_plan.query_resources} and {discovery_plan.search_queries} on iteration {iter_num}.")
                if discovery_plan.expand_tools:
                    logger.info(f"Expanding tools: {discovery_plan.expand_tools}")

            if not discovery_plan.refinement_needed:
                break
            if self.max_iter is not None and iter_num >= self.max_iter:
                break

            # Refine context
            ctx = await self.context.snapshot()
            expanded_tools = self._get_tool_map(available_tools, expanded=True)
            collapsed_tools = self._get_tool_map(available_tools, expanded=False)
            refinement_plan = await self.analyser.analyse(
                goal=goal,
                pursuit_assessment=pursuit_assessment,
                workspace_segment=ctx.segments,
                expanded_tools=expanded_tools,
                collapsed_tools=collapsed_tools,
                discovery_analysis=discovery_plan.discovery_analysis,
            )
            logger.info(f"Excluded resources: {refinement_plan.exclude_resources} and properties: {[f'{item.uri}:{item.property}' for item in refinement_plan.exclude_properties]} on iteration {iter_num}.")
            await self._apply_refinement_plan(refinement_plan)

        return iter_num

    async def _apply_discovery_plan(self, plan: DiscoveryPlan):
        for query in plan.search_queries:
            await self.context.search_resources(query)
        
        for uri in plan.query_resources:
            await self.context.query_resource(uri)
        
        self.expanded_tools.update(plan.expand_tools)

    async def _apply_refinement_plan(self, plan: RefinementPlan):
        for uri in plan.exclude_resources:
            await self.context.exclude_resource(uri)
        
        for item in plan.exclude_properties:
            await self.context.exclude_property(item.uri, item.property)
        
        self.expanded_tools.difference_update(plan.collapse_tools)
        
        if plan.sorted_segments:
            await self.context.sort_resources(plan.sorted_segments)

    def _convert_to_orchestration_action(
            self, 
            action_decision: ActionDecision
    ) -> OperationPlan | Resolution:
        """Convert ActionDecision to orchestration type."""
        if action_decision.decision_type == "execute" and action_decision.execution:
            return OperationPlan(
                reason=action_decision.situation_analysis,
                tool=action_decision.execution.tool,
                parameters=action_decision.execution.params,
            )
        elif action_decision.decision_type == "finalize" and action_decision.finalization:
            return Resolution(
                reason=action_decision.situation_analysis,
                response=action_decision.finalization.response,
                status=action_decision.finalization.status,
            )
        else:
            return Resolution(
                reason="Invalid action decision",
                response="Unable to process the action decision.",
                status="failed"
            )

    def _get_tool_map(self, available_tools: dict[str, SchematicTool], *, expanded: bool) -> dict[str, SchematicTool]:
        """Return the subset of tools that are expanded or collapsed."""
        if expanded:
            return {name: tool for name, tool in available_tools.items() if name in self.expanded_tools}
        else:
            return {name: tool for name, tool in available_tools.items() if name not in self.expanded_tools}
