"""Tool for fetching resources by URI."""

import json
from typing import Any

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository


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
