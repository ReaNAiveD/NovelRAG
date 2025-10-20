import json
from dataclasses import dataclass, field

from novelrag.agent.steps import StepDefinition, StepOutcome
from novelrag.agent.tool import LLMToolMixin
from novelrag.llm.types import ChatLLM
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.repository import ResourceRepository
from novelrag.template import TemplateEnvironment


@dataclass
class ResourceSegment:
    """Represents a partially loaded resource with selective properties."""
    uri: str
    included_properties: set[str] = field(default_factory=set)
    excluded_properties: set[str] = field(default_factory=set)


@dataclass
class ContextWorkspace:
    """The evolving set of resource segments during iterative building."""
    segments: dict[str, ResourceSegment] = field(default_factory=dict)
    sorted_uris: list[str] = field(default_factory=list)
    excluded_uris: set[str] = field(default_factory=set)

    def filter_relationships(self, relationships: dict[str, str]) -> dict[str, str]:
        """Filter out relationships that are excluded in the workspace."""
        return {uri: desc for uri, desc in relationships.items() if uri not in self.excluded_uris}
    
    def filter_children_ids(self, base_uri: str, children_ids: list[str]) -> list[str]:
        """Filter out children IDs that are excluded in the workspace."""
        return [uri for uri in children_ids if f"{base_uri}/{uri}" not in self.excluded_uris]
    
    def sorted_segments(self) -> list[ResourceSegment]:
        """Get the list of resource segments in sorted order."""
        return [self.segments[uri] for uri in self.sorted_uris if uri in self.segments]
    
    def ensure_segment(self, uri: str) -> ResourceSegment:
        """Ensure a resource segment exists for the given URI."""
        if uri not in self.segments:
            self.segments[uri] = ResourceSegment(uri=uri)
            self.sorted_uris.append(uri)
        return self.segments[uri]
    
    def sort_segments(self, sorted_uris: list[str]):
        """Sort the resource segments based on a new order of URIs."""
        unmentioned_uris = [uri for uri in self.segments if uri not in sorted_uris]
        self.sorted_uris = [uri for uri in sorted_uris if uri in self.segments] + unmentioned_uris
    
    def reset_excluded(self):
        """Reset excluded URIs and properties."""
        self.excluded_uris.clear()
        for segment in self.segments.values():
            segment.excluded_properties.clear()


@dataclass
class SearchHistoryItem:
    query: str
    aspect: str | None
    uris: list[str]


class ResourceContext(LLMToolMixin):
    def __init__(self, resource_repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        super().__init__(template_env, chat_llm)
        self.search_limit = 5
        self.resource_repo = resource_repo
        self.workspace = ContextWorkspace()
        self.workspace.ensure_segment("/")
        self.search_history: list[SearchHistoryItem] = []
    
    async def build_segment_data(self, segment: ResourceSegment) -> dict | None:
        resource = await self.resource_repo.find_by_uri(segment.uri)
        if not resource:
            return None
        if isinstance(resource, list):
            child_ids = [aspect.name for aspect in resource]
            filtered_child_ids = self.workspace.filter_children_ids("", child_ids)
            return {
                "uri": segment.uri,
                "pending_properties": [],
                "included_data": {},
                "relations": {},
                "child_ids": {"aspects": filtered_child_ids},
            }
        elif isinstance(resource, ResourceAspect):
            data = resource.aspect_dict
            child_ids = [element.id for element in resource.root_elements]
            filtered_child_ids = self.workspace.filter_children_ids(segment.uri, child_ids)
            pending_properties = set(data.keys()) - segment.included_properties - segment.excluded_properties
            included_data = {k: v for k, v in data.items() if k in segment.included_properties}
            return {
                "uri": segment.uri,
                "pending_properties": list(pending_properties),
                "included_data": included_data,
                "relations": {},
                "child_ids": {"top_child_resources": filtered_child_ids},
            }
        else:
            data = resource.props
            child_ids = resource.flattened_child_ids
            filtered_child_ids = {key: self.workspace.filter_children_ids(segment.uri, ids) for key, ids in child_ids.items()}
            pending_properties = set(data.keys()) - segment.included_properties - segment.excluded_properties
            included_data = {k: v for k, v in data.items() if k in segment.included_properties}
            relations = self.workspace.filter_relationships({uri: " ".join(desc) for uri, desc in resource.relations.items()})
            return {
                "uri": segment.uri,
                "pending_properties": list(pending_properties),
                "included_data": included_data,
                "relations": relations,
                "child_ids": filtered_child_ids,
            }
    
    async def query_resource(self, uri: str):
        self.workspace.ensure_segment(uri)
    
    async def search_resources(self, query: str, aspect: str | None = None):
        results = await self.resource_repo.vector_search(query, aspect=aspect, limit=self.search_limit)
        for res in results:
            self.workspace.ensure_segment(res.element.uri)
        self.search_history.append(SearchHistoryItem(query=query, aspect=aspect, uris=[res.element.uri for res in results]))
    
    async def exclude_resource(self, uri: str):
        self.workspace.excluded_uris.add(uri)
    
    async def include_property(self, uri: str, property_name: str):
        segment = self.workspace.ensure_segment(uri)
        segment.included_properties.add(property_name)
        if property_name in segment.excluded_properties:
            segment.excluded_properties.remove(property_name)
    
    async def exclude_property(self, uri: str, property_name: str):
        segment = self.workspace.ensure_segment(uri)
        segment.excluded_properties.add(property_name)
        if property_name in segment.included_properties:
            segment.included_properties.remove(property_name)
    
    async def sort_resources(self, sorted_uris: list[str]):
        self.workspace.sort_segments(sorted_uris)
    
    async def refine(
            self,
            sorted_segments: list[str],
            query_resources: list[str] | None = None,
            exclude_resources: list[str] | None = None,
            include_properties: list[dict[str, str]] | None = None,
            exclude_properties: list[dict[str, str]] | None = None,
            search_queries: list[str] | None = None,
    ):
        """Execute LLM's refinement decisions.
        
        Args:
            sorted_segments: Ordered list of URIs for segment priority
            query_resources: URIs to add to workspace  
            exclude_resources: URIs to exclude from workspace
            include_properties: Properties to include as [{"uri": "...", "property": "..."}]
            exclude_properties: Properties to exclude as [{"uri": "...", "property": "..."}]
            search_queries: Semantic search queries to run
        """
        
        # Query new resources
        for uri in query_resources or []:
            await self.query_resource(uri)
        
        # Search for resources
        for query in search_queries or []:
            await self.search_resources(query)
        
        # Exclude resources  
        for uri in exclude_resources or []:
            await self.exclude_resource(uri)
        
        # Include properties (dict format only)
        for prop_action in include_properties or []:
            await self.include_property(prop_action["uri"], prop_action["property"])
        
        # Exclude properties (dict format only)
        for prop_action in exclude_properties or []:
            await self.exclude_property(prop_action["uri"], prop_action["property"])
        
        await self.sort_resources(sorted_segments)
    
    async def build_context_for_planning(
            self,
            goal: str,
            last_step: StepOutcome | None,
            completed_steps: list[StepOutcome],
            pending_steps: list[str]
    ) -> dict[str, list[str]]:
        """Build context for planning based on current workspace state.
        
        This method generates the final planning context from all included
        resource properties and relationships in the workspace.
        
        Returns:
            Dictionary mapping resource URIs to lists of context strings
        """
        return await self._generate_final_context()
    
    async def _generate_final_context(self) -> dict[str, list[str]]:
        """Generate final context from workspace segments."""
        
        context = {}
        for segment in self.workspace.sorted_segments():
            if segment_data := await self.build_segment_data(segment):
                if segment.uri not in context:
                    context[segment.uri] = []
                for property_name, property_value in segment_data["included_data"].items():
                    context[segment.uri].append(f"{property_name}: {property_value}")
                for rel_uri, rel_desc in segment_data["relations"].items():
                    context[segment.uri].append(f"Related to {rel_uri}: {rel_desc}")
        
        return context
    
    def reset_workspace(self):
        """Clear excluded properties (reset to pending) but keep included properties."""
        self.workspace.reset_excluded()
        self.search_history = []
