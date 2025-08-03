"""Resource-specific tools for querying and writing."""

import json
import pydantic
from typing import Any, AsyncGenerator

from novelrag.llm.types import ChatLLM
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.template import TemplateEnvironment
from novelrag.resource.operation import validate_op_json

from .tool import SchematicTool, ContextualTool, LLMToolMixin
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
                "It allows you to define the structure and metadata of a new aspect, including its name, path, and any additional fields required for your application."
    
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

    async def call(self, **kwargs) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Create a new aspect and return its metadata."""
        name = kwargs.get('name')
        description = kwargs.get('description')
        if not name or not description:
            yield self.error("Both 'name' and 'description' are required to create a new aspect.")
            return
        aspect_metadata = await self.initialize_aspect_metadata(name, description)
        aspect = self.repo.add_aspect(name, aspect_metadata)
        self.message(f"Aspect '{name}' created successfully.")
        self.output(json.dumps(aspect.context_dict, ensure_ascii=False))

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
        return "This tool is used to fetch a specific resource or aspect from the repository by its URI. " \
        "Use this tool when you have a known URI and want to retrieve its complete content and metadata. " \
        "Supports root URI (`/`), aspect URIs (e.g., `/aspect`), and individual resource URIs (e.g., `/aspect/{+resource_id}` or `/aspect/parent_id/child_id`)."
    
    @property
    def output_description(self) -> str | None:
        return "Returns the specific resource or aspect identified by the URI. " \
               "For root URI (`/`): Returns all aspects in the repository. " \
               "For aspect URIs (`/aspect`): Returns aspect metadata including name, path, children_keys, and a list of root elements. " \
               "For resource URIs (`/aspect/resource_id` or `/aspect/parent_id/child_id`): Returns the individual resource with hierarchical structure. " \
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

    async def call(self, **kwargs) -> AsyncGenerator[ToolOutput, bool | str | None]:
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
            yield self.error("No URI provided. Please provide a resource or aspect URI to fetch.")
            return

        resource = await self.repo.find_by_uri(uri)
        if not resource:
            yield self.error(f"Resource or aspect with URI {uri} not found in the repository.")
            return

        if isinstance(resource, ResourceAspect | DirectiveElement):
            yield self.output(json.dumps(resource.context_dict, ensure_ascii=False))
        elif isinstance(resource, list):
            yield self.output(json.dumps([item.aspect_dict for item in resource], ensure_ascii=False))


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
               "Each resource has a hierarchical URI structure: `/aspect/resource_id` or `/aspect/parent_id/child_id`. " \
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

    async def call(self, **kwargs) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Perform semantic search and return matching resources."""
        query = kwargs.get('query')
        aspect = kwargs.get('aspect')
        top_k = kwargs.get('top_k', 5)
        
        if not query:
            yield self.error("No query provided. Please provide a search query string.")
            return
        
        result = await self.repo.vector_search(query, aspect=aspect, limit=top_k)
        if not result:
            yield self.message(f"No resources found matching the query: '{query}'")
            return
        
        yield self.message(f"Found {len(result)} resources matching the query: '{query}'")
        for item in result:
            yield self.output(json.dumps(item.element.context_dict, ensure_ascii=False))


class ResourceWriteTool(LLMToolMixin, ContextualTool):
    """Tool for editing existing content in the resource repository."""
    
    def __init__(self, repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.content_proposers: list[ContentProposer] = []
        self.repo = repo
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def description(self):
        return "This tool is used to edit existing content in the resource repository. " \
        "It generates proposals based on current beliefs and context, sorts them, " \
        "selects one for editing, builds an operation, and applies it to the repository. " \
        "It also discovers chain updates and backlog items based on the operation."

    async def call(self, believes: list[str] | None = None, step_description: str | None = None, context: list[str] | None = None, tools: dict[str, str] | None = None) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Edit existing content and return the updated content."""
        if believes is None:
            believes = []
        if step_description is None or not step_description:
            yield self.error("No step description provided. Please provide a description of the current step.")
            return
        if context is None:
            context = []
        
        related_context = await self.context_filter(step_description, context)
        proposals = [await proposer.propose(believes, step_description, related_context) for proposer in self.content_proposers]
        proposals = [item for sublist in proposals for item in sublist]
        if not proposals:
            yield self.warning("No proposals available.")
            return
        
        yield self.message(f"Generated {len(proposals)} proposals based on current beliefs and context.")
        sorted_proposals = await self.sort_proposals([p.content for p in proposals])
        yield self.message("Finished sorting proposals.")
        
        if not sorted_proposals:
            yield self.error("No valid proposals after sort to edit.")
            return
        
        selected_proposal = await self.select_proposal(sorted_proposals)
        yield self.message(f"Selected proposal: {selected_proposal}")
        yield self.step_progress("proposal_selection", selected_proposal, "Selected proposal for editing.")
        
        write_request = await self.discover_write_request(step_description, selected_proposal, context)
        if write_request:
            # Build new steps for write requests using step decomposition
            write_steps = []
            for request in write_request:
                if ":" in request:
                    tool, description = request.split(":", 1)
                    write_steps.append({
                        'tool': tool.strip(),
                        'description': description.strip()
                    })
            if write_steps:
                yield self.step_decomposition(
                    steps=write_steps,
                    rationale=f"Write requests discovered for proposal: {selected_proposal[:100]}..."
                )
                return
        
        yield self.debug("No new write request generated from the selected proposal.")
        operation = await self.build_operation(selected_proposal, step_description, context)
        if not operation:
            yield self.error("Failed to build operation from the selected proposal.")
            return
        
        yield self.debug("Built operation successfully. Start Validation.")
        try:
            op = validate_op_json(operation)
        except pydantic.ValidationError as e:
            yield self.error(f"Operation validation failed: {e}")
            return
        
        yield self.message("Operation validated successfully. Preparing to apply.")
        # TODO: Request the user to confirm the operation
        undo = await self.repo.apply(op)
        yield self.message(f"Applied operation. Undo operation created: {undo}")
        # TODO: Push the undo operation to the undo queue
        
        updates = await self.discover_chain_updates(step_description, operation)
        if updates:
            yield self.message(f"Discovered {len(updates)} chain updates.")
            # Use step decomposition to create chain update steps
            chain_steps = []
            for update in updates:
                if ":" in update:
                    tool, description = update.split(":", 1)
                    chain_steps.append({
                        'tool': tool.strip(),
                        'description': description.strip()
                    })
            if chain_steps:
                yield self.step_decomposition(
                    steps=chain_steps,
                    rationale=f"Chain updates discovered from operation: {operation[:100]}..."
                )
        
        backlog = await self.dicover_backlog(step_description, operation)
        if backlog:
            yield self.message(f"Discovered {len(backlog)} backlog items.")
            for item in backlog:
                yield self.backlog(content=item, priority="normal")
    
    async def context_filter(self, step_description: str, context: list[str]) -> list[str]:
        return (await self.call_template(
            'context_filter.jinja2',
            step_description=step_description,
            context=context,
        )).splitlines()
    
    async def sort_proposals(self, proposals: list[str]) -> list[str]:
        return (await self.call_template(
            'sort_edit_proposals.jinja2',
            proposals=proposals,
        )).splitlines()
    
    async def select_proposal(self, proposals: list[str]) -> str:
        raise NotImplementedError("The logic to select a proposal will be implemented in the future.")
    
    async def discover_write_request(self, step_description: str, proposal: str, context: list[str]) -> list[str]:
        return (await self.call_template(
            'discover_write_request.jinja2',
            step_description=step_description,
            proposal=proposal,
            context=context,
        )).splitlines()
    
    async def ensure_not_in_context(self, proposal: str, dependency: str, context: list[str]) -> bool:
        return (await self.call_template(
            'ensure_not_in_context.jinja2',
            proposal=proposal,
            dependency=dependency,
            context=context,
        )).lower() in ['yes', 'true', '1']
    
    async def build_new_steps(self, step_description: str, proposal: str, context: list[str], dependencies: list[str]) -> list[str]:
        return (await self.call_template(
            'build_new_steps.jinja2',
            step_description=step_description,
            proposal=proposal,
            context=context,
            dependencies=dependencies,
        )).splitlines()
    
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
    
    async def discover_chain_updates(self, step_description: str, operation: str) -> list[str]:
        return (await self.call_template(
            'discover_chain_updates.jinja2',
            step_description=step_description,
            operation=operation,
        )).splitlines()
    
    async def dicover_backlog(self, step_description: str, operation: str) -> list[str]:
        return (await self.call_template(
            'discover_backlog.jinja2',
            step_description=step_description,
            operation=operation,
        )).splitlines()


class ResourceTypeCreateTool:
    """Tool for creating new resource types."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def create(self, data: dict) -> str:
        """Create a new resource and return its identifier."""
        raise NotImplementedError("Subclasses should implement this method.")
