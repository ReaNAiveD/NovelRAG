"""Resource-specific tools for querying and writing."""

import json
import pydantic
import random
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.template import TemplateEnvironment
from novelrag.resource.operation import validate_op_json
from .llm_content_proposer import LLMContentProposer
from .steps import StepDefinition

from .tool import SchematicTool, ContextualTool, LLMToolMixin, ToolRuntime
from .types import ToolOutput
from .proposals import ContentProposer


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
            name=name,
            description=description,
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
        return "This tool fetches a specific resource, aspect, or repository root by its URI. " \
            "Use this when you have a known URI and want to retrieve complete content and metadata. " \
            "Supports: root URI (/), aspect URIs (/{aspect}), and resource URIs (/{aspect}/{resource_id} or /{aspect}/{parent_id}/{child_id}). " \
            "The repository uses hierarchical structure: query root URI for all aspects, " \
            "aspect URIs for metadata and root elements, " \
            "and resource URIs for full hierarchical structure including sub-resources."

    @property
    def output_description(self) -> str | None:
        return "Returns the specific resource or aspect identified by the URI. " \
               "For root URI (`/`): Returns all aspects in the repository. " \
               "For aspect URIs (`/{aspect}`): Returns aspect metadata including name, path, children_keys, and a list of root elements. " \
               "For resource URIs (`/{aspect}/{resource_id}` or `/{aspect}/{parent_id}/{child_id}`): Returns the individual resource with hierarchical structure. " \
               "Use the URI to navigate between parent and child resources. " \
               "The `relations` field maps related resource URIs to human-readable relationship descriptions. " \
               "Child resources are listed by ID only - compose full child URIs by combining the parent URI with the child ID."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "The URI of the resource or aspect to retrieve. Use `/` for all aspects, `/aspect` for aspect metadata, or `/aspect/{+resource_id}` for individual resources."
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
        return "This tool is used to search for resources in the repository using semantic vector search. " \
        "It finds resources related to your query string using AI-powered similarity matching. " \
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
                    "description": "The search query string to find semantically similar resources."
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
            await runtime.error("No query provided. Please provide a search query string.")
            return self.error("No query provided. Please provide a search query string.")

        result = await self.repo.vector_search(query, aspect=aspect, limit=top_k)
        if not result:
            await runtime.message(f"No resources found matching the query: '{query}'")
            return self.result(json.dumps([], ensure_ascii=False))

        await runtime.message(f"Found {len(result)} resources matching the query: '{query}'")
        items = [item.element.context_dict for item in result]
        return self.result(json.dumps(items, ensure_ascii=False))


class ResourceWriteTool(LLMToolMixin, ContextualTool):
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
    def description(self):
        return "This tool is used to edit existing content in the resource repository. " \
        "It generates proposals based on current beliefs and context, sorts them, " \
        "selects one for editing, builds an operation, and applies it to the repository. " \
        "It also discovers chain updates and backlog items based on the operation. " \
        "Before using this tool, ensure you have a clear understanding of the aspect you are going to edit, " \
        "you should follow the guidelines and limitations according to the aspect, " \
        "usually this is through the ResourceFetchTool or ResourceSearchTool."

    async def call(self, runtime: ToolRuntime, believes: list[str], step: StepDefinition, context: list[str], tools: dict[str, str] | None = None) -> ToolOutput:
        """Edit existing content and return the updated content."""
        if step.intent is None or not step.intent:
            return self.error("No step description provided. Please provide a description of the current step.")

        if cached_aspect := step.progress.get("aspect"):
            await runtime.message(f"resuming from cached proposal selection: {cached_aspect}")
            aspect = cached_aspect[0]
        else:
            aspect = await self._determine_target_aspect(step, context)
        if not aspect:
            return self.error("Could not determine target aspect for editing. Please provide a valid aspect name in the step definition.")
        await runtime.progress("aspect", aspect, f"Determined target aspect for editing: {aspect}")
        if self.repo.get_aspect(aspect) is None:
            await runtime.warning("The determined aspect does not exist in the repository. Decompose the step to create it first.")
            return self.decomposition(
                steps=[{
                    'description': f"Create the missing aspect '{aspect}' required for this editing operation.",
                    'context': f"The step '{step.intent}' requires aspect '{aspect}' which does not exist in the repository."
                }],
                rerun=True
            )

        proposal_sets = [await proposer.propose(believes, step.intent, context) for proposer in self.content_proposers]
        proposals = [item for sublist in proposal_sets for item in sublist]
        if not proposals:
            return self.error("No generated proposals available.")

        await runtime.message(f"Generated {len(proposals)} proposals based on current believes and context.")
        await runtime.debug(f"Proposals: {proposals}")
        sorted_proposals = await self._rank_proposals([p.content for p in proposals])
        await runtime.message("Finished sorting proposals.")

        if not sorted_proposals:
            return self.error("No valid proposals after sort to edit.")

        selected_proposal = await self.select_proposal(sorted_proposals)
        await runtime.message(f"Selected proposal: {selected_proposal}")
        await runtime.progress("proposal_selection", selected_proposal, "Selected proposal for editing.")

        await runtime.debug("No new write request generated from the selected proposal.")
        operation = await self.build_operation(selected_proposal, step.intent, context)
        if not operation:
            return self.error("Failed to build operation from the selected proposal.")

        await runtime.debug("Built operation successfully. Start Validation.")
        try:
            op = validate_op_json(operation)
        except pydantic.ValidationError as e:
            return self.error("Operation validation failed: " + str(e))

        await runtime.message("Operation validated successfully. Preparing to apply.")
        if not await runtime.confirmation(f"Do you want to apply the operation?\n{json.dumps(op, indent=2)}"):
            await runtime.message("Operation application cancelled by user.")
            return self.result("Operation application cancelled by user.")
        undo = await self.repo.apply(op)
        await runtime.message(f"Applied operation. Undo operation created: {undo}")
        # TODO: Push the undo operation to the undo queue

        # Discover required updates that need to be applied as triggered actions
        required_updates = await self._discover_required_updates(step.intent, operation, json.dumps(undo), context)

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
        backlog = await self._discover_backlog(step.intent, operation, json.dumps(undo), context)
        if backlog:
            await runtime.message(f"Discovered {len(backlog)} backlog items.")
            for item in backlog:
                priority = item.get("priority", "normal")
                await runtime.backlog(content=item, priority=priority)

        # Return the operation JSON that was applied as the result
        return self.result(json.dumps(operation))

    async def _context_filter(self, step_description: str, context: list[str]) -> list[str]:
        return (await self.call_template(
            'context_filter.jinja2',
            step_description=step_description,
            context=context,
        )).splitlines()

    async def _determine_target_aspect(self, step: StepDefinition, context: list[str]) -> str:
        aspects = await self.repo.all_aspects()

        response = await self.call_template(
            'determine_suitable_aspect.jinja2',
            json_format=True,
            step_intent=step.intent,
            context=context,
            aspects=[aspect.context_dict for aspect in aspects]
        )

        result = json.loads(response)
        return result.get('selected_aspect')

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

    async def build_operation(self, proposal: str, step_description: str, context: list[str]) -> str:
        return await self.call_template(
            'build_operation.jinja2',
            proposal=proposal,
            step_description=step_description,
            json_format=True
        )

    async def rebuild_operation(self, proposal: str, step_description: str, incorrect_op: str) -> str:
        return await self.call_template(
            'rebuild_operation.jinja2',
            proposal=proposal,
            step_description=step_description,
            incorrect_op=incorrect_op,
            json_format=True
        )

    async def _discover_required_updates(self, step_description: str, operation: str, undo_operation: str, context: list[str]) -> dict[str, list[dict[str, str]]]:
        """Discover cascade content updates and relation updates that need to be applied immediately."""
        response = await self.call_template(
            'discover_required_updates.jinja2',
            step_description=step_description,
            operation=operation,
            undo_operation=undo_operation,
            context=context,
            json_format=True
        )
        return json.loads(response)
    
    async def _discover_backlog(self, step_description: str, operation: str, undo_operation: str, context: list[str]) -> list[dict[str, Any]]:
        """Discover backlog items including dependency items and other future work items."""
        response = await self.call_template(
            'discover_backlog.jinja2',
            step_description=step_description,
            operation=operation,
            undo_operation=undo_operation,
            context=context,
            json_format=True
        )
        return json.loads(response)

    async def _find_aspect_creation_tool(self, aspect: str, step_intent: str, tools: dict[str, str]) -> dict[str, Any] | None:
        """Find a suitable tool to create the missing aspect using LLM analysis."""
        response = await self.call_template(
            'find_aspect_creation_tool.jinja2',
            json_format=True,
            aspect=aspect,
            step_intent=step_intent,
            tools=tools
        )

        result = json.loads(response)
        if result.get("found"):
            return {
                "tool_name": result.get("tool_name"),
                "rationale": result.get("rationale", "Tool found for aspect creation")
            }
        return None

    async def _parse_resource_dependencies(self, dependency: str, original_proposal: str) -> tuple[str, str]:
        """Parse the dependency string to extract aspect and description."""
        existing_aspects: list[ResourceAspect] = await self.repo.all_aspects()
        aspects_dict = {aspect.name: aspect.context_dict for aspect in existing_aspects}

        response = await self.call_template(
            'parse_resource_dependencies.jinja2',
            json_format=True,
            dependency=dependency,
            original_proposal=original_proposal,
            aspects=aspects_dict
        )

        result = json.loads(response)
        return result.get["aspect"], result.get("description", dependency)


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
            existing_relation = source_resource.relations.get(target_resource_uri)
            updated_relations = await self.get_updated_relations(
                source_resource, target_resource, existing_relation or [], operation, relation_description
            )
            await self.repo.update_relations(source_resource_uri, target_resource_uri, updated_relations)
            await runtime.message(f"Updated relations for source resource '{source_resource_uri}' to target resource '{target_resource_uri}'.")
            if isinstance(target_resource, DirectiveElement):
                existing_relation = target_resource.relations.get(source_resource_uri)
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
