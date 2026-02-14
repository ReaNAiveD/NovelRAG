"""Resource workspace management for context-driven orchestration.

This module provides classes for managing the dynamic resource context
used during orchestration:
- ResourceSegment: Partially loaded resource with selective properties
- SegmentData: Enriched view of a resource segment for LLM consumption
- ContextWorkspace: Evolving set of resource segments
- ResourceContext: High-level context management with search and query
- SearchHistoryItem: Tracks search queries and results
"""

from dataclasses import dataclass, field
from typing import Any

from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.repository import ResourceRepository


@dataclass
class ResourceSegment:
    """Represents a partially loaded resource with selective properties.
    
    This is the workspace's internal state tracker. Use SegmentData for
    the enriched view suitable for LLM/template consumption.
    """
    uri: str
    excluded_properties: set[str] = field(default_factory=set)


@dataclass
class SegmentData:
    """Enriched view of a resource segment for LLM consumption.
    
    Built from a ResourceSegment by loading its resource data,
    applying property exclusions, and resolving children/relations.
    """
    uri: str
    included_data: dict[str, Any]
    excluded_properties: list[str]
    child_ids: dict[str, list[str]]
    relations: dict[str, str]


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
        return [self.segments[uri] for uri in self.sorted_uris if uri in self.segments and uri not in self.excluded_uris]
    
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


class ResourceContext:
    """High-level resource context management for orchestration.
    
    Provides methods for:
    - Querying resources by URI
    - Semantic search across resources
    - Excluding resources/properties from context
    - Building context dictionaries for LLM consumption
    """
    
    def __init__(self, resource_repo: ResourceRepository):
        self.search_limit = 5
        self.resource_repo = resource_repo
        self.workspace = ContextWorkspace()
        self.workspace.ensure_segment("/")
        self.search_history: list[SearchHistoryItem] = []
    
    async def build_segment_data(self, segment: ResourceSegment) -> SegmentData | None:
        """Build an enriched view of a resource segment.
        
        Returns None if the resource does not exist.
        """
        resource = await self.resource_repo.find_by_uri(segment.uri)
        if not resource:
            return None
        if isinstance(resource, list):
            child_ids = [aspect.name for aspect in resource]
            filtered_child_ids = self.workspace.filter_children_ids("", child_ids)
            return SegmentData(
                uri=segment.uri,
                included_data={},
                excluded_properties=sorted(segment.excluded_properties),
                child_ids={"aspects": filtered_child_ids},
                relations={},
            )
        elif isinstance(resource, ResourceAspect):
            data = resource.aspect_dict
            child_ids = [element.id for element in resource.root_elements]
            filtered_child_ids = self.workspace.filter_children_ids(segment.uri, child_ids)
            included_properties = set(data.keys()) - segment.excluded_properties
            included_data = {k: v for k, v in data.items() if k in included_properties}
            return SegmentData(
                uri=segment.uri,
                included_data=included_data,
                excluded_properties=sorted(segment.excluded_properties),
                child_ids={"top_child_resources": filtered_child_ids},
                relations={},
            )
        else:
            data = resource.props
            child_ids = resource.flattened_child_ids
            filtered_child_ids = {key: self.workspace.filter_children_ids(segment.uri, ids) for key, ids in child_ids.items()}
            included_properties = set(data.keys()) - segment.excluded_properties
            included_data = {k: v for k, v in data.items() if k in included_properties}
            relations = self.workspace.filter_relationships({uri: " ".join(desc) for uri, desc in resource.relationships.items()})
            return SegmentData(
                uri=segment.uri,
                included_data=included_data,
                excluded_properties=sorted(segment.excluded_properties),
                child_ids=filtered_child_ids,
                relations=relations,
            )
    
    async def build_workspace_view(self) -> tuple[list[SegmentData], list[str]]:
        """Build enriched segment views for all loaded resources.
        
        Returns:
            A tuple of (segment_data_list, nonexistent_uris).
        """
        segments: list[SegmentData] = []
        nonexisted: list[str] = []
        for segment in self.workspace.sorted_segments():
            data = await self.build_segment_data(segment)
            if data:
                segments.append(data)
            else:
                nonexisted.append(segment.uri)
        return segments, nonexisted
    
    async def query_resource(self, uri: str):
        self.workspace.ensure_segment(uri)
    
    async def search_resources(self, query: str, aspect: str | None = None):
        results = await self.resource_repo.vector_search(query, aspect=aspect, limit=self.search_limit)
        for res in results:
            self.workspace.ensure_segment(res.element.uri)
        self.search_history.append(SearchHistoryItem(query=query, aspect=aspect, uris=[res.element.uri for res in results]))
    
    async def exclude_resource(self, uri: str):
        self.workspace.excluded_uris.add(uri)
    
    async def exclude_property(self, uri: str, property_name: str):
        segment = self.workspace.ensure_segment(uri)
        segment.excluded_properties.add(property_name)
    
    async def sort_resources(self, sorted_uris: list[str]):
        self.workspace.sort_segments(sorted_uris)
    
    async def dict_context(self) -> dict[str, list[str]]:
        """Generate final context from workspace segments."""
        context = {}
        segments, _ = await self.build_workspace_view()
        for segment_data in segments:
            if segment_data.uri not in context:
                context[segment_data.uri] = []
            for property_name, property_value in segment_data.included_data.items():
                context[segment_data.uri].append(f"{property_name}: {property_value}")
            for rel_uri, rel_desc in segment_data.relations.items():
                context[segment_data.uri].append(f"Related to {rel_uri}: {rel_desc}")
        return context

    def reset_workspace(self):
        """Clear excluded properties (reset to pending) but keep included properties."""
        self.workspace.reset_excluded()
        self.search_history = []
