"""Tool for editing existing content in the resource repository."""

import json
import pydantic
import random
from typing import Any

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.exceptions import OperationError
from novelrag.llm import LLMMixin
from langchain_core.language_models import BaseChatModel
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.resource.operation import validate_op
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment

from .types import ContentGenerationTask
from ..workspace import ResourceContext
from ..proposals import ContentProposer
from ..llm_content_proposer import LLMContentProposer


class ResourceWriteTool(LLMMixin, SchematicTool):
    """Tool for editing existing content in the resource repository."""
    
    def __init__(self, repo: ResourceRepository, context: ResourceContext, template_env: TemplateEnvironment, chat_llm: BaseChatModel,
                 backlog: Backlog[BacklogEntry] | None = None, undo_queue: UndoQueue | None = None):
        self.content_proposers: list[ContentProposer] = [LLMContentProposer(template_env=template_env, chat_llm=chat_llm)]
        self.context = context
        self.repo = repo
        self.backlog = backlog
        self.undo = undo_queue
        super().__init__(template_env=template_env, chat_llm=chat_llm)

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
                context=await self.context.dict_context()
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
        operations_json = await self.build_operations(
            action=operation_specification,
            context=await self.context.dict_context(),
            content_results=content_results
        )
        
        if not operations_json:
            return self.error("Failed to build operations from generated content.")

        await runtime.debug("Built operation successfully. Start Validation.")
        try:
            # Parse the operation JSON and extract all operations
            operations = json.loads(operations_json)['operations']
            
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
            context=await self.context.dict_context(),
        )

        # Process perspective updates first (higher priority)
        if required_updates.get("perspective_updates"):
            await runtime.message(f"Discovered {len(required_updates['perspective_updates'])} perspective updates.")
            for update in required_updates["perspective_updates"]:
                await runtime.debug(f"Perspective update: {update['reason']} - {update['content']}")
                operation = await self._build_perspective_update_operation(
                    update, 
                    step_description=operation_specification,
                    operations=[op.model_dump() for op in operations],
                    undo_operations=[op.model_dump() for op in undo_operations],
                    context=await self.context.dict_context(),
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
        if required_updates.get("relation_updates"):
            await runtime.message(f"Discovered {len(required_updates['relation_updates'])} relation updates.")
            for update in required_updates["relation_updates"]:
                await runtime.debug(f"Relation update: {update['reason']} - {update['content']}")
                await self._apply_relation_update(
                    runtime=runtime,
                    update=update,
                    step_description=operation_specification,
                    operations=[op.model_dump() for op in operations],
                    undo_operations=[op.model_dump() for op in undo_operations],
                    context=await self.context.dict_context(),
                )

        # Discover future work items for the backlog
        backlog = await self._discover_backlog(
            step_description=operation_specification,
            operations=[op.model_dump() for op in operations],
            undo_operations=[op.model_dump() for op in undo_operations],
            context=await self.context.dict_context(),
        )
        if backlog and self.backlog is not None:
            await runtime.message(f"Discovered {len(backlog)} backlog items.")
            for item in backlog:
                self.backlog.add_entry(BacklogEntry.from_dict(item))

        return self.result(json.dumps([op.model_dump() for op in operations], ensure_ascii=False))

    async def _rank_proposals(self, proposals: list[str]) -> list[str]:
        response = await self.call_template(
            'sort_edit_proposals.jinja2',
            proposals=proposals,
            json_format=True,
        )

        try:
            # Parse the JSON response
            result = json.loads(response)
            sorted_proposals = result.get('sorted_proposals', [])

            # Extract proposals in rank order and map back to original content
            ordered_proposals = []
            for item in sorted(sorted_proposals, key=lambda x: x.get('rank', float('inf'))):
                proposal_number = item.get('proposal_number')
                if proposal_number is not None and 1 <= proposal_number <= len(proposals):
                    # proposal_number is 1-based, convert to 0-based index
                    ordered_proposals.append(proposals[proposal_number - 1])

            return ordered_proposals

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Fallback to original behavior if JSON parsing fails
            return proposals

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

    async def build_operations(self, action: str, context: dict[str, list[str]], content_results: list[dict] | None = None) -> str:
        """Build operations from action description and optional content results.
        
        Args:
            action: Action description or operation specification
            context: Available context for reference
            content_results: Optional list of content generation results
            
        Returns:
            JSON string containing the operations
        """
        return await self.call_template(
            'build_operation.jinja2',
            action=action,
            context=context,
            content_results=content_results or [],
            json_format=True
        )

    async def _discover_required_updates(self, step_description: str, operations: list[dict], undo_operations: list[dict], context: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
        """Discover cascade content updates and relation updates that need to be applied immediately."""
        response = await self.call_template(
            'discover_required_updates.jinja2',
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
            json_format=True
        )
        return json.loads(response)
    
    async def _discover_backlog(self, step_description: str, operations: list[dict], undo_operations: list[dict], context: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Discover backlog items including dependency items and other future work items."""
        response = await self.call_template(
            'discover_backlog.jinja2',
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
            json_format=True
        )
        return json.loads(response)["backlog_items"]
    
    async def _build_perspective_update_operation(self, update: dict[str, str], step_description: str, operations: list[dict], undo_operations: list[dict], context: dict[str, list[str]]) -> dict:
        """Build a perspective update operation from the update description."""
        response = await self.call_template(
            'build_perspective_update_operation.jinja2',
            update=update,
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
            json_format=True
        )
        return json.loads(response)

    async def _apply_relation_update(
        self,
        runtime: ToolRuntime,
        update: dict[str, str],
        step_description: str,
        operations: list[dict],
        undo_operations: list[dict],
        context: dict[str, list[str]],
    ) -> None:
        """Apply a relation update to both sides of the relationship."""
        # Parse URIs from the update content
        uris = await self._parse_relation_update_uris(update, context)
        source_uri = uris.get('source_uri')
        target_uri = uris.get('target_uri')
        
        if not source_uri or not target_uri:
            await runtime.warning(f"Could not parse relation update URIs: {uris.get('error', 'Unknown error')}")
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
            update=update,
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
            source_to_target_relations = updated_relations.get('source_to_target_relations', [])
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
            target_to_source_relations = updated_relations.get('target_to_source_relations', [])
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

    async def _parse_relation_update_uris(self, update: dict[str, str], context: dict[str, list[str]]) -> dict[str, str | None]:
        """Parse source and target URIs from a relation update description."""
        response = await self.call_template(
            'parse_relation_update_uris.jinja2',
            update=update,
            context=context,
            json_format=True
        )
        return json.loads(response)

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
        context: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Build updated relation lists for both directions."""
        response = await self.call_template(
            'build_relation_update.jinja2',
            update=update,
            source_resource=source_resource,
            target_resource=target_resource,
            source_to_target_existing=source_to_target_existing,
            target_to_source_existing=target_to_source_existing,
            step_description=step_description,
            operations=operations,
            undo_operations=undo_operations,
            context=context,
            json_format=True
        )
        return json.loads(response)
