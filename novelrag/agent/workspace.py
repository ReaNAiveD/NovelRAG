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
                "child_ids": {"top_resources": filtered_child_ids},
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
    
    async def build_context_for_execution(self, tool: str | None, intent: str, threshold: int = 50, max_iterations: int = 5) -> dict[str, list[str]]:
        """Main iterative context building loop for step execution."""
        
        for iteration in range(max_iterations):
            # Build current workspace state for LLM
            workspace_segments = []
            nonexisted_uris = []
            for segment in self.workspace.sorted_segments():
                if segment_data := await self.build_segment_data(segment):
                    workspace_segments.append(segment_data)
                else:
                    nonexisted_uris.append(segment.uri)
            
            # Get LLM refinement decisions
            response = await self.call_template(
                "refine_context_for_execution.jinja2",
                tool=tool or "inference",
                intent=intent,
                workspace_segments=workspace_segments,
                nonexisted_uris=nonexisted_uris,
                search_history=[{"query": item.query, "aspect": item.aspect, "uris": item.uris} for item in self.search_history],
                threshold=threshold,
                json_format=True
            )
            
            decisions = json.loads(response)
            
            # Execute refinement decisions
            await self._execute_refinement_decisions(decisions)
            
            # Check completion
            if decisions.get("refinement_complete", False):
                break
        
        # Generate final context
        return await self._generate_final_context()
    
    async def build_context_for_planning(self, goal: str, last_step: StepOutcome | None = None, completed_steps: list[StepOutcome] | None = None, pending_steps: list[StepDefinition] | None = None, threshold: int = 50, max_iterations: int = 5) -> dict[str, list[str]]:
        """Main iterative context building loop for planning decisions."""
        for iteration in range(max_iterations):
            # Build current workspace state for LLM
            workspace_segments = []
            nonexisted_uris = []
            for segment in self.workspace.sorted_segments():
                if segment_data := await self.build_segment_data(segment):
                    workspace_segments.append(segment_data)
                else:
                    nonexisted_uris.append(segment.uri)

            # Get LLM refinement decisions
            response = await self.call_template(
                "refine_context_for_planning.jinja2",
                goal=goal,
                last_step={
                    "tool": last_step.action.tool,
                    "intent": last_step.action.intent,
                    "status": last_step.status.value,
                    "result": "\n".join(last_step.results),
                    "error": last_step.error_message,
                } if last_step else None,
                completed_steps=[{
                    "tool": step.action.tool,
                    "intent": step.action.intent,
                    "status": step.status.value,
                    "result": "\n".join(step.results),
                    "error": step.error_message,
                } for step in completed_steps] if completed_steps else [],
                pending_steps=[{
                    "tool": step.tool,
                    "intent": step.intent,
                } for step in pending_steps] if pending_steps else [],
                workspace_segments=workspace_segments,
                excluded_uris=list(self.workspace.excluded_uris),
                nonexisted_uris=nonexisted_uris,
                search_history=[{"query": item.query, "aspect": item.aspect, "uris": item.uris} for item in self.search_history],
                threshold=threshold,
                json_format=True
            )
            
            decisions = json.loads(response)
            
            # Execute refinement decisions
            await self._execute_refinement_decisions(decisions)
            
            # Check completion
            if decisions.get("refinement_complete", False):
                break
        
        # Generate final context
        return await self._generate_final_context()
    
    async def _execute_refinement_decisions(self, decisions: dict):
        """Execute LLM's refinement decisions."""
        
        # Query new resources
        for uri in decisions.get("query_resources", []):
            await self.query_resource(uri)
        
        # Search for resources
        for query in decisions.get("search_queries", []):
            await self.search_resources(query)
        
        # Exclude resources  
        for uri in decisions.get("exclude_resources", []):
            await self.exclude_resource(uri)
        
        # Include properties
        for prop_action in decisions.get("include_properties", []):
            await self.include_property(prop_action["uri"], prop_action["property"])
        
        # Exclude properties
        for prop_action in decisions.get("exclude_properties", []):
            await self.exclude_property(prop_action["uri"], prop_action["property"])
        
        # Resort segments (mandatory in each iteration)
        if "sorted_segments" in decisions:
            await self.sort_resources(decisions["sorted_segments"])
    
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
