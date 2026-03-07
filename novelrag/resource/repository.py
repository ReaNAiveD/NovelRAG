from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any

from .aspect import ResourceAspect
from .element import Element
from .operation import Operation

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    distance: float
    element: Element


@dataclass
class RemovedAspectResult:
    """Data returned by ``remove_aspect`` to support full undo/redo.

    Attributes:
        aspect: The removed ResourceAspect object.
        restore_context: Backend-specific restoration data.
        elements: Element dicts for restoring via ``add_aspect(elements=...)``.
                  Structure is backend-specific (nested for local, flat for PG).
    """
    aspect: ResourceAspect
    restore_context: dict[str, Any]
    elements: list[dict]


class ResourceRepository(ABC):
    @abstractmethod
    async def all_aspects(self) -> list[ResourceAspect]:
        pass

    @abstractmethod
    async def get_aspect(self, name: str) -> ResourceAspect | None:
        pass

    @abstractmethod
    async def add_aspect(self, name: str, metadata: dict[str, Any], elements: list[dict] | None = None) -> ResourceAspect:
        """Add a new aspect to the repository.

        Args:
            name: The name of the aspect.
            metadata: Aspect configuration fields (``path``, ``children_keys``, etc.).
            elements: Optional list of nested element dicts to restore.  When
                      provided the elements are loaded into the LUT and vector
                      store and the aspect YAML file is written to disk.
        """
        pass

    @abstractmethod
    async def remove_aspect(self, name: str) -> RemovedAspectResult | None:
        """Remove an aspect and all its elements from the repository.

        Returns a :class:`RemovedAspectResult` containing the aspect, its
        elements, and backend-specific restore context so the operation can
        be reversed, or ``None`` if the aspect does not exist.
        """
        pass

    @abstractmethod
    async def iter_elements(self, aspect_name: str) -> list[Element]:
        """Return every element in *aspect_name* (flat list)."""
        pass

    @abstractmethod
    async def find_by_uri(self, resource_uri: str) -> list[str] | ResourceAspect | Element | None:
        """Find a resource by its URI in the repository.
        
        Args:
            resource_uri: The URI of the resource to find
        
        Returns:
            - list[str]: All aspect names if resource_uri is '/'
            - ResourceAspect: Single aspect if resource_uri matches '/{aspect_name}'
            - Element: Element if found by URI
            - None: If no match is found
        """
        pass

    @abstractmethod
    async def vector_search(self, query: str, *, aspect: str | None = None, limit: int | None = None) -> list[SearchResult]:
        """Search for resources using vector similarity.
        
        Args:
            query: The search query string
            aspect: Optional aspect to filter results
            limit: Maximum number of results to return
        """
        pass

    @abstractmethod
    async def apply(self, op: Operation) -> Operation:
        """Apply an operation to modify the repository.
        
        Args:
            op: The operation to apply
        """
        pass

    @abstractmethod
    async def update_relationships(self, source_uri: str, target_uri: str, relationships: list[str]) -> list[str]:
        """Update the relationships of a resource by its URI.
        Args:
            source_uri: The URI of the resource to update
            target_uri: The URI of the target resource to relate to
            relationships: A dictionary of relations to set for the resource
        Returns:
            List[str]: The old relationships before the update
        """
        pass
