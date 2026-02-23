"""Pipelines extracted from ResourceWriteTool.

Each pipeline encapsulates a reusable, multi-step process that orchestrates
LLM calls and environment interaction through an ``ExecutionContext``.

* ``ContentGenerationProcedure`` – generate, rank, and select content proposals.
* ``CascadeUpdateProcedure`` – discover and apply perspective + relation cascade updates.
* ``BacklogDiscoveryProcedure`` – discover future work items for the backlog.
"""

import json
import random
from typing import Annotated

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from novelrag.agenturn.procedure import ExecutionContext
from novelrag.exceptions import OperationError
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.resource.operation import validate_op
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.resource_agent.workspace import ResourceContext, ContextSnapshot
from novelrag.resource_agent.propose import ContentProposer
from novelrag.resource_agent.tool.types import ContentGenerationTask
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm


# ---------------------------------------------------------------------------
# Shared LLM response models
# ---------------------------------------------------------------------------


class RankedProposal(BaseModel):
    """A single proposal with its rank."""
    rank: Annotated[int, Field(description="Rank position (1 = best).")]
    proposal_number: Annotated[int, Field(description="1-based index of the proposal in the original list.")]


class RankProposalsResponse(BaseModel):
    """LLM response for ranking content proposals."""
    sorted_proposals: Annotated[list[RankedProposal], Field(
        default_factory=list,
        description="Proposals sorted by quality.",
    )]


class CascadeUpdate(BaseModel):
    """A single cascade update item with reason and content."""
    reason: Annotated[str, Field(description="Brief explanation of why this update is needed.")]
    content: Annotated[str, Field(description="Natural language description of what needs to be updated.")]


class DiscoverRequiredUpdatesResponse(BaseModel):
    """LLM response for discovering cascade updates."""
    perspective_updates: Annotated[list[CascadeUpdate], Field(
        default_factory=list,
        description="Content updates that should be applied immediately.",
    )]
    relation_updates: Annotated[list[CascadeUpdate], Field(
        default_factory=list,
        description="Relation updates that should be applied immediately.",
    )]


class BacklogItem(BaseModel):
    """A single backlog work item with concrete typed fields."""
    type: Annotated[str, Field(description="Category of the item (e.g. 'dependency', 'character_development').")]
    priority: Annotated[str, Field(description="Priority level: 'high', 'normal', or 'low'.")]
    description: Annotated[str, Field(description="Human-readable description of the work to do.")]
    context_reference: Annotated[str, Field(default="", description="Reference to the part of the operation that created this item.")]
    search_guidance: Annotated[str, Field(default="", description="Instructions for finding existing compatible resources (dependency items).")]
    creation_guidance: Annotated[str, Field(default="", description="Instructions for creating if no existing resource fits (dependency items).")]
    aspect_hint: Annotated[str, Field(default="", description="Suggested aspect type for the resource (dependency items).")]
    rationale: Annotated[str, Field(default="", description="Why this backlog item is important (general items).")]
    target_resources: Annotated[str, Field(default="", description="URIs or descriptions of resources this work would affect (general items).")]


class DiscoverBacklogResponse(BaseModel):
    """LLM response for discovering future work items."""
    backlog_items: Annotated[list[BacklogItem], Field(
        default_factory=list,
        description="Future work items to add to the backlog.",
    )]


class ParseRelationUrisResponse(BaseModel):
    """LLM response for parsing relation update URIs."""
    source_uri: Annotated[str | None, Field(default=None, description="Parsed source resource URI.")]
    target_uri: Annotated[str | None, Field(default=None, description="Parsed target resource URI.")]
    error: Annotated[str | None, Field(default=None, description="Error message if URIs could not be parsed.")]


class BuildRelationUpdateResponse(BaseModel):
    """LLM response for building updated relation lists."""
    source_to_target_relations: Annotated[list[str], Field(
        default_factory=list,
        description="Updated relation descriptions from source to target.",
    )]
    target_to_source_relations: Annotated[list[str], Field(
        default_factory=list,
        description="Updated relation descriptions from target to source.",
    )]


# ---------------------------------------------------------------------------
# ContentGenerationProcedure
# ---------------------------------------------------------------------------


class ContentGenerationProcedure:
    """Generate, rank, and select content for a set of tasks.

    This **Procedure** orchestrates multiple LLM calls (content proposal,
    ranking) and interacts with the environment through an
    ``ExecutionContext``.
    """

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    SORT_PROPOSALS_TEMPLATE = "sort_edit_proposals.jinja2"

    def __init__(
        self,
        content_proposers: list[ContentProposer],
        chat_llm: BaseChatModel,
        context: ResourceContext,
        lang: str = "en",
        lang_directive: str = "",
    ):
        self._proposers = content_proposers
        self._context = context
        self._lang_directive = lang_directive
        self._rank_llm = chat_llm.with_structured_output(RankProposalsResponse)
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._sort_proposals_tmpl = template_env.load_template(self.SORT_PROPOSALS_TEMPLATE)

    async def execute(
        self,
        operation_specification: str,
        content_generation_tasks: list[ContentGenerationTask],
        ctx: ExecutionContext,
    ) -> list[dict]:
        """Generate content for each task and return the results.

        Returns a list of dicts with keys ``description``, ``content_key``,
        and ``content``.
        """
        content_results: list[dict] = []
        for i, task in enumerate(content_generation_tasks):
            await ctx.info(f"Generating content for task {i+1}/{len(content_generation_tasks)}: {task.description}")
            await ctx.output(f"Generating content: {task.description}")

            content_description = (
                f"Generate content for the following task:\n"
                f"Task: {task.description}\n\n"
                f"This content is part of a broader operation: {operation_specification}\n\n"
                f"Ensure the generated content aligns with both the specific task requirements "
                f"and the overall operation goal."
            )

            proposals = [await proposer.propose(
                believes=[],
                content_description=content_description,
                context=await self._context.snapshot()
            ) for proposer in self._proposers]
            proposals = [proposal for proposal_set in proposals for proposal in proposal_set]
            if not proposals:
                await ctx.warning(f"No content generated for task: {task.description}")
                continue
            ranked_proposals = await self._rank_proposals([proposal.content for proposal in proposals])
            selected_content = await self._select_proposal(ranked_proposals)
            await ctx.debug(f"Generated content: {selected_content}")

            content_results.append({
                "description": task.description,
                "content_key": task.content_key,
                "content": selected_content
            })

        return content_results

    @trace_llm("proposal_ranking")
    async def _rank_proposals(self, proposals: list[str]) -> list[str]:
        prompt = self._sort_proposals_tmpl.render(proposals=proposals)
        response = await self._rank_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Rank the proposals by quality."),
        ])
        assert isinstance(response, RankProposalsResponse)

        ordered_proposals = []
        for item in sorted(response.sorted_proposals, key=lambda x: x.rank):
            if 1 <= item.proposal_number <= len(proposals):
                ordered_proposals.append(proposals[item.proposal_number - 1])

        return ordered_proposals if ordered_proposals else proposals

    @staticmethod
    async def _select_proposal(proposals: list[str]) -> str:
        """Select a proposal using weighted random selection."""
        if not proposals:
            raise ValueError("Cannot select from empty proposals list")
        if len(proposals) == 1:
            return proposals[0]
        weights = [2**(len(proposals) - 1 - i) for i in range(len(proposals))]
        return random.choices(proposals, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# CascadeUpdateProcedure
# ---------------------------------------------------------------------------


class CascadeUpdateProcedure:
    """Discover and apply cascade updates (perspective + relation).

    This **Procedure** discovers required updates after a primary operation,
    then applies perspective updates and relation updates through the
    repository.
    """

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    DISCOVER_UPDATES_TEMPLATE = "discover_required_updates.jinja2"
    BUILD_PERSPECTIVE_TEMPLATE = "build_perspective_update_operation.jinja2"
    PARSE_RELATION_URIS_TEMPLATE = "parse_relation_update_uris.jinja2"
    BUILD_RELATION_TEMPLATE = "build_relation_update.jinja2"

    def __init__(
        self,
        repo: ResourceRepository,
        chat_llm: BaseChatModel,
        context: ResourceContext,
        lang: str = "en",
        lang_directive: str = "",
        undo_queue: UndoQueue | None = None,
    ):
        self._repo = repo
        self._context = context
        self._chat_llm = chat_llm
        self._lang_directive = lang_directive
        self._undo = undo_queue
        self._discover_updates_llm = chat_llm.with_structured_output(DiscoverRequiredUpdatesResponse)
        self._parse_uris_llm = chat_llm.with_structured_output(ParseRelationUrisResponse)
        self._relation_update_llm = chat_llm.with_structured_output(BuildRelationUpdateResponse)

        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._discover_updates_tmpl = template_env.load_template(self.DISCOVER_UPDATES_TEMPLATE)
        self._build_perspective_tmpl = template_env.load_template(self.BUILD_PERSPECTIVE_TEMPLATE)
        self._parse_relation_uris_tmpl = template_env.load_template(self.PARSE_RELATION_URIS_TEMPLATE)
        self._build_relation_tmpl = template_env.load_template(self.BUILD_RELATION_TEMPLATE)

    async def execute(
        self,
        step_description: str,
        applied_operations: list[dict],
        undo_operations: list[dict],
        ctx: ExecutionContext,
    ) -> tuple[list[dict], list[dict]]:
        """Discover and apply cascade updates.

        Returns ``(perspective_updates_applied, relation_updates_applied)``
        where each is a list of dicts describing what was changed.
        """
        required_updates = await self._discover_required_updates(
            step_description=step_description,
            operations=applied_operations,
            undo_operations=undo_operations,
            context=await self._context.snapshot(),
        )

        perspective_updates_applied: list[dict] = []
        relation_updates_applied: list[dict] = []

        # Process perspective updates first (higher priority)
        if required_updates.perspective_updates:
            await ctx.output(f"Discovered {len(required_updates.perspective_updates)} perspective cascade update(s).")
            for update in required_updates.perspective_updates:
                await ctx.debug(f"Perspective update: {update.reason} - {update.content}")
                await ctx.output(f"Updating perspective: {update.content}")
                operation = await self._build_perspective_update_operation(
                    update,
                    step_description=step_description,
                    operations=applied_operations,
                    undo_operations=undo_operations,
                    context=await self._context.snapshot(),
                )
                validated_operation = validate_op(operation)
                await ctx.info(f"Applying perspective update operation: {validated_operation}")
                try:
                    undo_op = await self._repo.apply(validated_operation)
                    if self._undo is not None:
                        self._undo.add_undo_item(ReversibleAction(method="apply", params={"op": undo_op.model_dump()}), clear_redo=True)
                    perspective_updates_applied.append({
                        "reason": update.reason,
                        "content": update.content,
                        "operation": validated_operation.model_dump(),
                        "undo_operation": undo_op.model_dump(),
                    })
                except OperationError as e:
                    await ctx.warning(f"Failed to apply perspective update operation: {e}\nOperation: {validated_operation}")

        # Process relation updates second
        if required_updates.relation_updates:
            await ctx.output(f"Discovered {len(required_updates.relation_updates)} relationship cascade update(s).")
            for update in required_updates.relation_updates:
                await ctx.debug(f"Relation update: {update.reason} - {update.content}")
                await ctx.output(f"Updating relationship: {update.content}")
                rel_result = await self._apply_relation_update(
                    ctx=ctx,
                    update=update,
                    step_description=step_description,
                    operations=applied_operations,
                    undo_operations=undo_operations,
                    context=await self._context.snapshot(),
                )
                if rel_result is not None:
                    relation_updates_applied.append(rel_result)

        return perspective_updates_applied, relation_updates_applied

    @trace_llm("discover_updates")
    async def _discover_required_updates(self, step_description: str, operations: list[dict], undo_operations: list[dict], context: ContextSnapshot) -> DiscoverRequiredUpdatesResponse:
        """Discover cascade content updates and relation updates."""
        prompt = self._discover_updates_tmpl.render(
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )
        response = await self._discover_updates_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Discover required updates."),
        ])
        assert isinstance(response, DiscoverRequiredUpdatesResponse)
        return response

    @trace_llm("perspective_update")
    async def _build_perspective_update_operation(self, update: CascadeUpdate, step_description: str, operations: list[dict], undo_operations: list[dict], context: ContextSnapshot) -> dict:
        """Build a perspective update operation from the update description."""
        prompt = self._build_perspective_tmpl.render(
            update=update.model_dump(),
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )
        response = await self._chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Build the perspective update operation in JSON format."),
        ], response_format={"type": "json_object"})
        assert isinstance(response.content, str), "Expected string content from LLM response"
        return json.loads(response.content)

    async def _apply_relation_update(
        self,
        ctx: ExecutionContext,
        update: CascadeUpdate,
        step_description: str,
        operations: list[dict],
        undo_operations: list[dict],
        context: ContextSnapshot,
    ) -> dict | None:
        """Apply a relation update to both sides of the relationship."""
        uris = await self._parse_relation_update_uris(update.model_dump(), context)
        source_uri = uris.source_uri
        target_uri = uris.target_uri

        if not source_uri or not target_uri:
            await ctx.warning(f"Could not parse relation update URIs: {uris.error or 'Unknown error'}")
            return None

        source_resource = await self._repo.find_by_uri(source_uri)
        target_resource = await self._repo.find_by_uri(target_uri)

        if not source_resource:
            await ctx.warning(f"Source resource not found: {source_uri}")
            return None
        if not target_resource:
            await ctx.warning(f"Target resource not found: {target_uri}")
            return None
        if isinstance(source_resource, list):
            await ctx.warning(f"Source URI '{source_uri}' points to multiple resources")
            return None
        if isinstance(target_resource, list):
            await ctx.warning(f"Target URI '{target_uri}' points to multiple resources")
            return None

        source_to_target_existing: list[str] = []
        target_to_source_existing: list[str] = []
        if isinstance(source_resource, DirectiveElement):
            source_to_target_existing = source_resource.relationships.get(target_uri, [])
        if isinstance(target_resource, DirectiveElement):
            target_to_source_existing = target_resource.relationships.get(source_uri, [])

        updated_relations = await self._build_relation_update(
            update=update.model_dump(),
            source_resource=source_resource.context_dict if hasattr(source_resource, 'context_dict') else {},
            target_resource=target_resource.context_dict if hasattr(target_resource, 'context_dict') else {},
            source_to_target_existing=source_to_target_existing,
            target_to_source_existing=target_to_source_existing,
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )

        rel_result: dict = {
            "source_uri": source_uri,
            "target_uri": target_uri,
            "old_source_to_target": source_to_target_existing,
            "new_source_to_target": source_to_target_existing,
            "old_target_to_source": target_to_source_existing,
            "new_target_to_source": target_to_source_existing,
        }

        if isinstance(source_resource, DirectiveElement):
            source_to_target_relations = updated_relations.source_to_target_relations
            old_relationships = await self._repo.update_relationships(source_uri, target_uri, source_to_target_relations)
            if self._undo is not None:
                self._undo.add_undo_item(ReversibleAction(
                    method="update_relationships",
                    params={"source_uri": source_uri, "target_uri": target_uri, "relations": old_relationships}
                ), clear_redo=True)
            await ctx.output(f"Updated relations: {source_uri} → {target_uri}")
            rel_result["old_source_to_target"] = old_relationships
            rel_result["new_source_to_target"] = source_to_target_relations

        if isinstance(target_resource, DirectiveElement):
            target_to_source_relations = updated_relations.target_to_source_relations
            old_relationships = await self._repo.update_relationships(target_uri, source_uri, target_to_source_relations)
            if self._undo is not None:
                self._undo.add_undo_item(ReversibleAction(
                    method="update_relationships",
                    params={"source_uri": target_uri, "target_uri": source_uri, "relations": old_relationships}
                ), clear_redo=True)
            await ctx.output(f"Updated relations: {target_uri} → {source_uri}")
            rel_result["old_target_to_source"] = old_relationships
            rel_result["new_target_to_source"] = target_to_source_relations

        return rel_result

    @trace_llm("parse_relation_uris")
    async def _parse_relation_update_uris(self, update: dict[str, str], context: ContextSnapshot) -> ParseRelationUrisResponse:
        """Parse source and target URIs from a relation update description."""
        prompt = self._parse_relation_uris_tmpl.render(update=update, context=context)
        response = await self._parse_uris_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Parse the relation URIs."),
        ])
        assert isinstance(response, ParseRelationUrisResponse)
        return response

    @trace_llm("build_relation_update")
    async def _build_relation_update(
        self,
        update: dict[str, str],
        source_resource: dict,
        target_resource: dict,
        source_to_target_existing: list[str],
        target_to_source_existing: list[str],
        step_description: str,
        operations: list[dict],
        undo_operations: list[dict],
        context: ContextSnapshot,
    ) -> BuildRelationUpdateResponse:
        """Build updated relation lists for both directions."""
        prompt = self._build_relation_tmpl.render(
            update=update,
            source_resource=source_resource,
            target_resource=target_resource,
            source_to_target_existing=source_to_target_existing,
            target_to_source_existing=target_to_source_existing,
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )
        response = await self._relation_update_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Build the relation update."),
        ])
        assert isinstance(response, BuildRelationUpdateResponse)
        return response


# ---------------------------------------------------------------------------
# BacklogDiscoveryProcedure
# ---------------------------------------------------------------------------


class BacklogDiscoveryProcedure:
    """Discover future work items for the backlog.

    This **Procedure** analyses completed operations and discovers follow-up
    work items that should be tracked in the backlog.
    """

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    DISCOVER_BACKLOG_TEMPLATE = "discover_backlog.jinja2"

    def __init__(
        self,
        chat_llm: BaseChatModel,
        context: ResourceContext,
        lang: str = "en",
        lang_directive: str = "",
        backlog: Backlog[BacklogEntry] | None = None,
    ):
        self._context = context
        self._backlog = backlog
        self._lang_directive = lang_directive
        self._discover_backlog_llm = chat_llm.with_structured_output(DiscoverBacklogResponse)
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._discover_backlog_tmpl = template_env.load_template(self.DISCOVER_BACKLOG_TEMPLATE)

    async def execute(
        self,
        step_description: str,
        applied_operations: list[dict],
        undo_operations: list[dict],
        ctx: ExecutionContext,
    ) -> int:
        """Discover and store backlog items. Returns the count of items added."""
        backlog_items = await self._discover_backlog(
            step_description=step_description,
            operations=applied_operations,
            undo_operations=undo_operations,
            context=await self._context.snapshot(),
        )
        backlog_count = 0
        if backlog_items and self._backlog is not None:
            backlog_count = len(backlog_items)
            await ctx.output(f"Discovered {backlog_count} backlog item(s):\n{''.join(f'- {item.description}\n' for item in backlog_items)}")
            for item in backlog_items:
                self._backlog.add_entry(BacklogEntry.from_dict(item.model_dump()))
        return backlog_count

    @trace_llm("discover_backlog")
    async def _discover_backlog(self, step_description: str, operations: list[dict], undo_operations: list[dict], context: ContextSnapshot) -> list[BacklogItem]:
        """Discover backlog items including dependency items and other future work items."""
        prompt = self._discover_backlog_tmpl.render(
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )
        response = await self._discover_backlog_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Discover backlog items."),
        ])
        assert isinstance(response, DiscoverBacklogResponse)
        return response.backlog_items
