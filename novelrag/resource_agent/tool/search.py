"""Tool for searching resources using semantic vector search."""

import json
from typing import Any

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.resource.repository import ResourceRepository


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
            await runtime.info(f"No resources found matching the query: '{query}'")
            return self.result(json.dumps([], ensure_ascii=False))

        await runtime.info(f"Found {len(result)} resources matching the query: '{query}'")
        items = [item.element.context_dict for item in result]
        return self.result(json.dumps(items, ensure_ascii=False))
