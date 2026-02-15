"""Tool for editing existing content in the resource repository."""

import json
import pydantic
import random
from typing import Annotated, Any

from pydantic import BaseModel, Field

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.exceptions import OperationError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.resource.operation import validate_op
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm

from novelrag.resource_agent.tool.types import ContentGenerationTask
from novelrag.resource_agent.workspace import ResourceContext, ContextSnapshot
from novelrag.resource_agent.propose import ContentProposer, LLMContentProposer


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


class ResourceWriteTool(SchematicTool):
    """Tool for editing existing content in the resource repository."""

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    SORT_PROPOSALS_TEMPLATE = "sort_edit_proposals.jinja2"
    BUILD_OPERATION_TEMPLATE = "build_operation.jinja2"
    DISCOVER_UPDATES_TEMPLATE = "discover_required_updates.jinja2"
    DISCOVER_BACKLOG_TEMPLATE = "discover_backlog.jinja2"
    BUILD_PERSPECTIVE_TEMPLATE = "build_perspective_update_operation.jinja2"
    PARSE_RELATION_URIS_TEMPLATE = "parse_relation_update_uris.jinja2"
    BUILD_RELATION_TEMPLATE = "build_relation_update.jinja2"

    def __init__(self, repo: ResourceRepository, context: ResourceContext, chat_llm: BaseChatModel,
                 lang: str = "en", backlog: Backlog[BacklogEntry] | None = None, undo_queue: UndoQueue | None = None):
        self.content_proposers: list[ContentProposer] = [LLMContentProposer(chat_llm=chat_llm, lang=lang)]
        self.context = context
        self.repo = repo
        self.backlog = backlog
        self.undo = undo_queue
        self.chat_llm = chat_llm  # kept for Tier 3 calls (build_operations, _build_perspective_update_operation)

        # Structured-output LLM wrappers
        self._rank_llm = chat_llm.with_structured_output(RankProposalsResponse)
        self._discover_updates_llm = chat_llm.with_structured_output(DiscoverRequiredUpdatesResponse)
        self._discover_backlog_llm = chat_llm.with_structured_output(DiscoverBacklogResponse)
        self._parse_uris_llm = chat_llm.with_structured_output(ParseRelationUrisResponse)
        self._relation_update_llm = chat_llm.with_structured_output(BuildRelationUpdateResponse)

        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._sort_proposals_tmpl = template_env.load_template(self.SORT_PROPOSALS_TEMPLATE)
        self._build_operation_tmpl = template_env.load_template(self.BUILD_OPERATION_TEMPLATE)
        self._discover_updates_tmpl = template_env.load_template(self.DISCOVER_UPDATES_TEMPLATE)
        self._discover_backlog_tmpl = template_env.load_template(self.DISCOVER_BACKLOG_TEMPLATE)
        self._build_perspective_tmpl = template_env.load_template(self.BUILD_PERSPECTIVE_TEMPLATE)
        self._parse_relation_uris_tmpl = template_env.load_template(self.PARSE_RELATION_URIS_TEMPLATE)
        self._build_relation_tmpl = template_env.load_template(self.BUILD_RELATION_TEMPLATE)

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def prerequisites(self) -> str | None:
        return "Requires the target aspect to already exist in the repository. Use AspectCreateTool first if the aspect is missing."
    
    @property
    def description(self):
        return "Use this tool to modify existing resources within established aspects. " \
        "This tool plans and executes repository operations based on your specified intent. " \
        "Supports comprehensive operations including creating/updating/deleting resources, " \
        "modifying properties, splicing element lists (insert/remove/replace), " \
        "flattening hierarchies, and merging resources. " \
        "Automatically handles cascading updates to related resources and discovers follow-up work for your backlog."
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation_specification": {
                    "type": "string",
                    "description": "Detailed natural language description of the operation to be performed on existing aspects. Include target resource URIs (e.g., /character/john), specific fields to update, and any relationships to consider. Be specific about what exactly needs to be done, include all fields/properties that need updates, mention any relationships or dependencies, and use natural language but be precise."
                },
                "content_generation_tasks": {
                    "type": "array",
                    "description": "List of content generation tasks to perform as part of the operation. Break down into focused, specific tasks where each task should generate content for a clear purpose. All available context will be used for content generation.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "What content to generate for this specific task. Should be focused and specific, generating content for a clear purpose."
                            },
                            'content_key': {
                                'type': 'string',
                                'description': 'Key identifier for organizing this content in the resource structure (optional). Used when building the final resource object from multiple generated content pieces.'
                            }
                        },
                        "required": ["description"]
                    }
                }
            },
            "required": ["operation_specification", "content_generation_tasks"]
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Edit existing content using the new planning-based workflow."""
        operation_specification = kwargs.get('operation_specification')
        if not operation_specification:
            return self.error("No operation specification provided. Please provide a detailed description of the operation to perform.")
        content_generation_tasks = kwargs.get('content_generation_tasks', [])
        if not content_generation_tasks:
            return self.error("No content generation tasks provided. Please provide at least one content generation task.")
        content_generation_tasks = [ContentGenerationTask(**task) for task in content_generation_tasks]

        await runtime.message(f"Operation planned: {operation_specification}")
        await runtime.message(f"Content generation tasks: {len(content_generation_tasks)}")

        content_results = []
        for i, task in enumerate(content_generation_tasks):
            await runtime.message(f"Generating content for task {i+1}/{len(content_generation_tasks)}: {task.description}")

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
                context=await self.context.snapshot()
            ) for proposer in self.content_proposers]
            proposals = [proposal for proposal_set in proposals for proposal in proposal_set]
            if not proposals:
                await runtime.warning(f"No content generated for task: {task.description}")
                continue
            ranked_proposals = await self._rank_proposals([proposal.content for proposal in proposals])
            selected_content = await self.select_proposal(ranked_proposals)
            await runtime.debug(f"Generated content: {selected_content}")

            content_results.append({
                "description": task.description,
                "content_key": task.content_key,
                "content": selected_content
            })

        if not content_results:
            return self.error("No content was generated for any task.")

        await runtime.message(f"Generated content for {len(content_results)} tasks")

        await runtime.message("Building operations from generated content...")
        try:
            operations = await self.build_operations(
                action=operation_specification,
                context=await self.context.snapshot(),
                content_results=content_results
            )            
            # Validate all operations first
            operations = [validate_op(op) for op in operations]
        except pydantic.ValidationError as e:
            return self.error("Operation validation failed: " + str(e))

        await runtime.message("Operation validated successfully. Preparing to apply.")
        if not await runtime.confirmation(f"Do you want to apply {len(operations)} operation(s)?\n{json.dumps([op.model_dump() for op in operations], indent=2, ensure_ascii=False)}"):
            await runtime.message("Operation application cancelled by user.")
            return self.result("Operation application cancelled by user.")
        undo_operations = [await self.repo.apply(op) for op in operations][::-1]
        await runtime.message(f"Applied operation. Undo operation created: {undo_operations}")
        if self.undo is not None:
            for undo_op in undo_operations:
                self.undo.add_undo_item(ReversibleAction(method="apply", params={"op": undo_op.model_dump()}), clear_redo=True)

        # Discover required updates that need to be applied as triggered actions
        required_updates = await self._discover_required_updates(
            step_description=operation_specification,
            operations=[op.model_dump() for op in operations],
            undo_operations=[op.model_dump() for op in undo_operations],
            context=await self.context.snapshot(),
        )

        # Process perspective updates first (higher priority)
        if required_updates.perspective_updates:
            await runtime.message(f"Discovered {len(required_updates.perspective_updates)} perspective updates.")
            for update in required_updates.perspective_updates:
                await runtime.debug(f"Perspective update: {update.reason} - {update.content}")
                operation = await self._build_perspective_update_operation(
                    update, 
                    step_description=operation_specification,
                    operations=[op.model_dump() for op in operations],
                    undo_operations=[op.model_dump() for op in undo_operations],
                    context=await self.context.snapshot(),
                )
                validated_operation = validate_op(operation)
                await runtime.message(f"Applying perspective update operation: {validated_operation}")
                try:
                    undo_op = await self.repo.apply(validated_operation)
                    if self.undo is not None:
                        self.undo.add_undo_item(ReversibleAction(method="apply", params={"op": undo_op.model_dump()}), clear_redo=True)
                except OperationError as e:
                    await runtime.warning(f"Failed to apply perspective update operation: {e}\nOperation: {validated_operation}")

        # Process relation updates second
        if required_updates.relation_updates:
            await runtime.message(f"Discovered {len(required_updates.relation_updates)} relation updates.")
            for update in required_updates.relation_updates:
                await runtime.debug(f"Relation update: {update.reason} - {update.content}")
                await self._apply_relation_update(
                    runtime=runtime,
                    update=update,
                    step_description=operation_specification,
                    operations=[op.model_dump() for op in operations],
                    undo_operations=[op.model_dump() for op in undo_operations],
                    context=await self.context.snapshot(),
                )

        # Discover future work items for the backlog
        backlog = await self._discover_backlog(
            step_description=operation_specification,
            operations=[op.model_dump() for op in operations],
            undo_operations=[op.model_dump() for op in undo_operations],
            context=await self.context.snapshot(),
        )
        if backlog and self.backlog is not None:
            await runtime.message(f"Discovered {len(backlog)} backlog items.")
            for item in backlog:
                self.backlog.add_entry(BacklogEntry.from_dict(item.model_dump()))

        return self.result(json.dumps([op.model_dump() for op in operations], ensure_ascii=False))

    @trace_llm("proposal_ranking")
    async def _rank_proposals(self, proposals: list[str]) -> list[str]:
        prompt = self._sort_proposals_tmpl.render(proposals=proposals)
        response = await self._rank_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Rank the proposals by quality."),
        ])
        assert isinstance(response, RankProposalsResponse)

        ordered_proposals = []
        for item in sorted(response.sorted_proposals, key=lambda x: x.rank):
            if 1 <= item.proposal_number <= len(proposals):
                ordered_proposals.append(proposals[item.proposal_number - 1])

        return ordered_proposals if ordered_proposals else proposals

    async def select_proposal(self, proposals: list[str]) -> str:
        """Select a proposal using weighted random selection.

        Earlier indices have exponentially higher probability of being selected.
        """
        if not proposals:
            raise ValueError("Cannot select from empty proposals list")

        if len(proposals) == 1:
            return proposals[0]

        # Create weights where each index has half the weight of the previous index
        # Weight for index i = 2^(n-1-i) where n is the total number of proposals
        weights = [2**(len(proposals) - 1 - i) for i in range(len(proposals))]
        # Use random.choices for weighted random selection
        selected_proposal = random.choices(proposals, weights=weights, k=1)[0]
        return selected_proposal

    @trace_llm("build_operations")
    async def build_operations(self, action: str, context: ContextSnapshot, content_results: list[dict] | None = None) -> list[dict]:
        """Build operations from action description and optional content results.
        
        Args:
            action: Action description or operation specification
            context: Available context for reference
            content_results: Optional list of content generation results
            
        Returns:
            JSON string containing the operations
        """
        prompt = self._build_operation_tmpl.render(
            action=action,
            context=context,
            content_results=content_results or [],
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate the operations in JSON format."),
        ], response_format={"type": "json_object"})
        assert isinstance(response.content, str), "Expected string content from LLM response"
        return json.loads(response.content)["operations"]

    @trace_llm("discover_updates")
    async def _discover_required_updates(self, step_description: str, operations: list[dict], undo_operations: list[dict], context: ContextSnapshot) -> DiscoverRequiredUpdatesResponse:
        """Discover cascade content updates and relation updates that need to be applied immediately."""
        prompt = self._discover_updates_tmpl.render(
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
        )
        response = await self._discover_updates_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Discover required updates."),
        ])
        assert isinstance(response, DiscoverRequiredUpdatesResponse)
        return response
    
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
            SystemMessage(content=prompt),
            HumanMessage(content="Discover backlog items."),
        ])
        assert isinstance(response, DiscoverBacklogResponse)
        return response.backlog_items
    
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
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Build the perspective update operation in JSON format."),
        ], response_format={"type": "json_object"})
        assert isinstance(response.content, str), "Expected string content from LLM response"
        return json.loads(response.content)

    async def _apply_relation_update(
        self,
        runtime: ToolRuntime,
        update: CascadeUpdate,
        step_description: str,
        operations: list[dict],
        undo_operations: list[dict],
        context: ContextSnapshot,
    ) -> None:
        """Apply a relation update to both sides of the relationship."""
        # Parse URIs from the update content
        uris = await self._parse_relation_update_uris(update.model_dump(), context)
        source_uri = uris.source_uri
        target_uri = uris.target_uri
        
        if not source_uri or not target_uri:
            await runtime.warning(f"Could not parse relation update URIs: {uris.error or 'Unknown error'}")
            return
        
        # Fetch resources
        source_resource = await self.repo.find_by_uri(source_uri)
        target_resource = await self.repo.find_by_uri(target_uri)
        
        if not source_resource:
            await runtime.warning(f"Source resource not found: {source_uri}")
            return
        if not target_resource:
            await runtime.warning(f"Target resource not found: {target_uri}")
            return
        if isinstance(source_resource, list):
            await runtime.warning(f"Source URI '{source_uri}' points to multiple resources")
            return
        if isinstance(target_resource, list):
            await runtime.warning(f"Target URI '{target_uri}' points to multiple resources")
            return
        
        # Get existing relations
        source_to_target_existing: list[str] = []
        target_to_source_existing: list[str] = []
        
        if isinstance(source_resource, DirectiveElement):
            source_to_target_existing = source_resource.relationships.get(target_uri, [])
        if isinstance(target_resource, DirectiveElement):
            target_to_source_existing = target_resource.relationships.get(source_uri, [])
        
        # Build updated relations using template
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

        # Apply source → target relations
        if isinstance(source_resource, DirectiveElement):
            source_to_target_relations = updated_relations.source_to_target_relations
            old_relationships = await self.repo.update_relationships(source_uri, target_uri, source_to_target_relations)
            if self.undo is not None:
                self.undo.add_undo_item(ReversibleAction(
                    method="update_relationships",
                    params={
                        "source_uri": source_uri,
                        "target_uri": target_uri,
                        "relations": old_relationships
                    }
                ), clear_redo=True)
            await runtime.message(f"Updated relations: {source_uri} → {target_uri}")
        
        # Apply target → source relations
        if isinstance(target_resource, DirectiveElement):
            target_to_source_relations = updated_relations.target_to_source_relations
            old_relationships = await self.repo.update_relationships(target_uri, source_uri, target_to_source_relations)
            if self.undo is not None:
                self.undo.add_undo_item(ReversibleAction(
                    method="update_relationships",
                    params={
                        "source_uri": target_uri,
                        "target_uri": source_uri,
                        "relations": old_relationships
                    }
                ), clear_redo=True)
            await runtime.message(f"Updated relations: {target_uri} → {source_uri}")

    @trace_llm("parse_relation_uris")
    async def _parse_relation_update_uris(self, update: dict[str, str], context: ContextSnapshot) -> ParseRelationUrisResponse:
        """Parse source and target URIs from a relation update description."""
        prompt = self._parse_relation_uris_tmpl.render(
            update=update,
            context=context,
        )
        response = await self._parse_uris_llm.ainvoke([
            SystemMessage(content=prompt),
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
            SystemMessage(content=prompt),
            HumanMessage(content="Build the relation update."),
        ])
        assert isinstance(response, BuildRelationUpdateResponse)
        return response
