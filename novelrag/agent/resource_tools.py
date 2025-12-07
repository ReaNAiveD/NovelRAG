"""Resource-specific tools for querying and writing."""

import json
import pydantic
import random
from dataclasses import dataclass, field
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.template import TemplateEnvironment
from novelrag.resource.operation import validate_op
from .llm_content_proposer import LLMContentProposer

from .tool import SchematicTool, LLMToolMixin, ToolRuntime
from .types import ToolOutput
from .proposals import ContentProposer


@dataclass
class ContentGenerationTask:
    """Represents a content generation task."""
    description: str                    # What content to generate
    content_key: str | None = None    # Which field this content will update


@dataclass
class OperationPlan:
    """Result of operation planning phase."""
    operation_specification: str       # Detailed natural language intent
    content_generation_tasks: list[ContentGenerationTask]
    missing_aspects: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)

    @property
    def has_prerequisites(self) -> bool:
        """Check if there are any unmet prerequisites."""
        return bool(self.missing_aspects or self.missing_context)

    def prerequisites_error(self) -> str:
        """Generate error message for unmet prerequisites."""
        errors = []
        if self.missing_aspects:
            errors.append(f"Missing aspects: {', '.join(self.missing_aspects)}")
        if self.missing_context:
            errors.append(f"Missing context: {', '.join(self.missing_context)}")
        return "; ".join(errors)


class AspectCreateTool(LLMToolMixin, SchematicTool):
    """Tool for creating new aspects in the resource repository."""
    
    def __init__(self, repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.repo = repo
        super().__init__(template_env=template_env, chat_llm=chat_llm)
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def description(self):
        return "This tool is used to create new aspects in the resource repository. " \
                "It allows you to define the structure and metadata of a new aspect, including its name, path, and any additional fields required for your application." \
                "Before using this tool, you should have a clear understanding of all other aspects in the repository, " \
                "as the new aspect will be added to the existing structure."
    
    @property
    def output_description(self) -> str | None:
        return "Returns the newly created aspect's metadata, including its name, path, and any additional fields defined during creation."
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the new aspect to create."
                },
                "description": {
                    "type": "string",
                    "description": "A brief description of the aspect's purpose and context."
                },
            },
            "required": ["name", "description"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Create a new aspect and return its metadata."""
        name = kwargs.get('name')
        description = kwargs.get('description')
        if not name or not description:
            await runtime.error("Both 'name' and 'description' are required to create a new aspect.")
            return self.error("You must provide both 'name' and 'description'.")
        aspect_metadata = await self.initialize_aspect_metadata(name, description)
        aspect = self.repo.add_aspect(name, aspect_metadata)
        await runtime.message(f"Aspect '{name}' created successfully.")
        return self.result(json.dumps(aspect.context_dict, ensure_ascii=False))

    async def initialize_aspect_metadata(self, name: str, description: list[str]) -> dict[str, Any]:
        return json.loads(await self.call_template(
            'initialize_aspect_metadata.jinja2',
            json_format=True,
            aspect_name=name,
            aspect_description=description,
        ))


class ResourceFetchTool(SchematicTool):
    """Tool for fetching a specific resource by its URI."""
    
    def __init__(self, repo: ResourceRepository):
        self.repo = repo

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def description(self):
        return "This tool fetches resources, aspects, or repository root by URI. " \
               "Supports hierarchical queries: root URI (`/`) returns all aspect names, " \
               "aspect URIs (`/{aspect}`) return aspect metadata with child resource names, " \
               "and resource URIs return individual resources with their child resource names if any. " \
               "Child URIs can be composed by appending `/{child_name}` to the parent URI. " \
               "Use returned names for subsequent targeted queries."

    @property
    def output_description(self) -> str | None:
        return "For root URI (`/`): Returns all aspect names in the repository. " \
               "For aspect URIs (`/{aspect}`): Returns aspect metadata including child resource names. " \
               "For resource URIs (`/{aspect}/{resource}` or deeper): Returns the individual resource with full content and child resource names if any. " \
               "Child URIs are composed by appending `/{child_name}` to the parent URI. " \
               "Use returned names to construct URIs for individual child queries. " \
               "The `relations` field maps related resource URIs to human-readable relationship descriptions."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "URIs follow Linux-style path format: "
                                   "Root: `/` \n"
                                   "Format: `/{aspect}/{resource_name}` (e.g., `/character/john_doe`) \n"
                                   "Hierarchical: `/{aspect}/{parent}/{resource_name}` (e.g., `/location/castle_main_hall/throne_room`) \n"
                                   "Multi-level: `/{aspect}/{grandparent}/{parent}/{resource_name}` for deeply nested resources \n"
                                   "NEVER use just names like 'John' - always use complete URIs like `/character/john_doe`"
                }
            },
            "required": ["uri"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Fetch a resource or aspect by URI and return its content.

        For Root URI ('/'): Returns all aspects in the repository.
    
        For aspect URIs (e.g., '/aspect'): Returns the aspect metadata including name, path, 
        children_keys, and a list of root elements.
        
        For resource URIs (e.g., '/aspect/resource_id' or '/aspect/parent_id/child_id'): 
        Returns the individual resource with its full hierarchical structure, including
        relations mapped to human-readable descriptions and child resource IDs.
        """
        uri = kwargs.get('uri')
        if not uri:
            await runtime.error(f"No URI provided. Please provide a resource or aspect URI to fetch.")
            return self.error(f"No URI provided. Please provide a resource or aspect URI to fetch.")

        resource = await self.repo.find_by_uri(uri)
        if not resource:
            await runtime.error(f"Resource or aspect with URI {uri} not found in the repository.")
            return self.error(f"Resource or aspect with URI {uri} not found in the repository.")

        if isinstance(resource, ResourceAspect | DirectiveElement):
            return self.result(json.dumps(resource.context_dict, ensure_ascii=False))
        elif isinstance(resource, list):
            return self.result(json.dumps([item.aspect_dict for item in resource], ensure_ascii=False))
        return self.error(f"Unexpected resource type for URI {uri}. Please check the URI and try again.")


class ResourceSearchTool(SchematicTool):
    """Tool for searching resources using semantic vector search."""
    
    def __init__(self, repo: ResourceRepository):
        self.repo = repo

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def description(self):
        return "This tool performs semantic search across resources using embedding vector similarity. " \
        "It finds resources related to your query based on meaning and context. " \
        "Optionally filter by aspect and control the number of results returned."
    
    @property
    def output_description(self) -> str | None:
        return "Returns a list of resources that are semantically similar to the search query, ordered by relevance. " \
               "Each resource has a hierarchical URI structure: `/{aspect}/{resource_id}` or `/{aspect}/{parent_id}/{child_id}`. " \
               "Use the URI to navigate between parent and child resources. " \
               "The `relations` field maps related resource URIs to human-readable relationship descriptions. " \
               "Child resources are listed by ID only - compose full child URIs by combining the parent URI with the child ID."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding semantically similar resources using embedding vector similarity."
                },
                "aspect": {
                    "type": "string",
                    "description": "Optional aspect to filter the search results to specific resource types."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Defaults to 5.",
                    "default": 5
                }
            },
            "required": ["query"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Perform semantic search and return matching resources."""
        query = kwargs.get('query')
        aspect = kwargs.get('aspect')
        top_k = kwargs.get('top_k', 5)
        
        if not query:
            return self.error("No query provided. Please provide a search query string.")

        result = await self.repo.vector_search(query, aspect=aspect, limit=top_k)
        if not result:
            await runtime.message(f"No resources found matching the query: '{query}'")
            return self.result(json.dumps([], ensure_ascii=False))

        await runtime.message(f"Found {len(result)} resources matching the query: '{query}'")
        items = [item.element.context_dict for item in result]
        return self.result(json.dumps(items, ensure_ascii=False))


class ResourceWriteTool(LLMToolMixin, SchematicTool):
    """Tool for editing existing content in the resource repository."""
    
    def __init__(self, repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: ChatLLM,
                 compatibility_threshold: float = 0.7):
        self.content_proposers: list[ContentProposer] = [LLMContentProposer(template_env=template_env, chat_llm=chat_llm)]
        self.repo = repo
        self.compatibility_threshold = compatibility_threshold
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

    @property
    def require_context(self) -> bool:
        """This tool requires context to operate effectively."""
        return True

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        """Edit existing content using the new planning-based workflow."""
        operation_specification = kwargs.get('operation_specification')
        if not operation_specification:
            return self.error("No operation specification provided. Please provide a detailed description of the operation to perform.")
        content_generation_tasks = kwargs.get('content_generation_tasks', [])
        if not content_generation_tasks:
            return self.error("No content generation tasks provided. Please provide at least one content generation task.")
        content_generation_tasks = [ContentGenerationTask(**task) for task in content_generation_tasks]
        context = kwargs.get('context', {})

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
                context=context  # Use all available context
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
            context=context,
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
        # TODO: Push the undo operation to the undo queue

        # Discover required updates that need to be applied as triggered actions
        required_updates = await self._discover_required_updates(
            step_description=operation_specification,
            operations=[op.model_dump() for op in operations],
            undo_operations=[op.model_dump() for op in undo_operations],
            context=context)

        # Process perspective updates first (higher priority)
        if required_updates.get("perspective_updates"):
            await runtime.message(f"Discovered {len(required_updates['perspective_updates'])} perspective updates.")
            for update in required_updates["perspective_updates"]:
                await runtime.trigger_action(update)
                await runtime.debug(f"Perspective update: {update['reason']} - {update['content']}")

        # Process relation updates second
        if required_updates.get("relation_updates"):
            await runtime.message(f"Discovered {len(required_updates['relation_updates'])} relation updates.")
            for update in required_updates["relation_updates"]:
                await runtime.trigger_action(update)
                await runtime.debug(f"Relation update: {update['reason']} - {update['content']}")

        # Discover future work items for the backlog
        backlog = await self._discover_backlog(
            step_description=operation_specification,
            operations=[op.model_dump() for op in operations],
            undo_operations=[op.model_dump() for op in undo_operations],
            context=context)
        if backlog:
            await runtime.message(f"Discovered {len(backlog)} backlog items.")
            for item in backlog:
                priority = item.get("priority", "normal")
                await runtime.backlog(content=item, priority=priority)

        # Return the operations JSON that was applied as the result
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


class ResourceRelationWriteTool(LLMToolMixin, SchematicTool):
    def __init__(self, repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.repo = repo
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return 'This tool is used to write relations between resources in the repository. ' \
                'It allows you to define relationships between resources, such as A do something with object B or A take part in event C. ' \
                'The tool will generate a proposal for the relationship and apply it to the repository.'

    @property
    def output_description(self) -> str | None:
        return 'Returns the updated resource with the new relationship applied.'

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_resource_uri": {
                    "type": "string",
                    "description": "The URI of the source resource to which the relation will be added."
                },
                "target_resource_uri": {
                    "type": "string",
                    "description": "The URI of the target resource to which the relation will be added."
                },
                "operation": {
                    "type": "string",
                    "description": "The operation to perform on the relation, e.g., 'add', 'remove', 'update'."
                },
                "relation_description": {
                    "type": "string",
                    "description": "A human-readable description of the relationship between the source and target resources."
                }
            },
            "required": ["source_resource_uri", "target_resource_uri", "operation", "relation_description"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        source_resource_uri = kwargs.get('source_resource_uri')
        target_resource_uri = kwargs.get('target_resource_uri')
        operation = kwargs.get('operation')
        relation_description = kwargs.get('relation_description')
        if not source_resource_uri or not target_resource_uri or not operation or not relation_description:
            await runtime.error("Missing required parameters: source_resource_uri, target_resource_uri, operation, relation_description.")
            return self.error("Missing required parameters: source_resource_uri, target_resource_uri, operation, relation_description.")
        source_resource = await self.repo.find_by_uri(source_resource_uri)
        target_resource = await self.repo.find_by_uri(target_resource_uri)
        if isinstance(target_resource, list):
            await runtime.error(f"Target resource URI '{target_resource_uri}' points to multiple resources. Please specify a single resource.")
            return self.error(f"Target resource URI '{target_resource_uri}' points to multiple resources. Please specify a single resource.")
        if not target_resource:
            await runtime.error(f"Target resource URI '{target_resource_uri}' not found in the repository.")
            return self.error(f"Target resource URI '{target_resource_uri}' not found in the repository.")
        if isinstance(source_resource, DirectiveElement):
            existing_relation = source_resource.relationships.get(target_resource_uri)
            updated_relations = await self.get_updated_relations(
                source_resource, target_resource, existing_relation or [], operation, relation_description
            )
            await self.repo.update_relations(source_resource_uri, target_resource_uri, updated_relations)
            await runtime.message(f"Updated relations for source resource '{source_resource_uri}' to target resource '{target_resource_uri}'.")
            if isinstance(target_resource, DirectiveElement):
                existing_relation = target_resource.relationships.get(source_resource_uri)
                updated_relations = await self.get_updated_relations(
                    target_resource, source_resource, existing_relation or [], operation, relation_description
                )
                await self.repo.update_relations(target_resource_uri, source_resource_uri, updated_relations)
                await runtime.message(f"Updated relations for target resource '{target_resource_uri}' to source resource '{source_resource_uri}'.")
            return self.result(json.dumps(source_resource.context_dict, ensure_ascii=False))
        else:
            await runtime.error(f"Source resource URI '{source_resource_uri}' does not point to a valid resource. Please check the URI.")
            return self.error(f"Source resource URI '{source_resource_uri}' does not point to a valid resource. Please check the URI.")

    async def get_updated_relations(self, source_resource: DirectiveElement, target_resource: DirectiveElement | ResourceAspect, existing_relation: list[str], operation: str, relation_description: str) -> list[str]:
        return json.loads(await self.call_template(
            'get_updated_relations.jinja2',
            json_format=True,
            source_resource=source_resource.context_dict,
            target_resource=target_resource.context_dict,
            existing_relation=existing_relation,
            operation=operation,
            relation_description=relation_description,
        ))['relations']
