import logging
from typing import Any

from novelrag.config.resource import AspectConfig

logger = logging.getLogger(__name__)


class ResourceAspect:
    """A named category that groups related elements.

    ``root_elements`` holds the *ordered* list of root-level element names
    (i.e. their ``id`` strings).  The application layer can reconstruct URIs
    from these names (``/{aspect_name}/{element_name}``).
    """

    def __init__(
        self,
        name: str,
        children_keys: list[str],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        root_element_names: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.children_keys = children_keys
        self.metadata = metadata or {}
        self.root_element_names: list[str] = root_element_names or []

    @classmethod
    def from_config(
        cls,
        name: str,
        config: AspectConfig,
        root_element_names: list[str] | None = None,
    ) -> "ResourceAspect":
        """Create a ResourceAspect from an AspectConfig object."""
        return cls(
            name=name,
            description=config.description,
            children_keys=config.children_keys,
            metadata=config.model_extra,
            root_element_names=root_element_names,
        )

    def to_config(self, path: str) -> AspectConfig:
        """Convert the ResourceAspect to an AspectConfig object."""
        return AspectConfig.model_validate(
            {
                "path": path,
                "children_keys": self.children_keys,
                **({"description": self.description} if self.description else {}),
                **self.metadata,  # Include any additional metadata as extra fields
            }
        )

    def update(self, metadata_updates: dict[str, Any]) -> dict[str, Any]:
        """Update the aspect's metadata with the provided updates.

        Returns a dictionary of the previous values for any keys that were updated,
        which can be used for undo operations.
        """
        undo_data = {}
        for key, value in metadata_updates.items():
            if key in ["name", "children_keys", "description", "path"]:
                logger.warning(f'Ignore Reserved metadata key "{key}" Update in aspect "{self.name}".')
            elif key in self.metadata and value is None:
                undo_data[key] = self.metadata[key]
                del self.metadata[key]
            elif value is not None:
                undo_data[key] = self.metadata.get(key)
                self.metadata[key] = value
        return undo_data

    @property
    def aspect_dict(self):
        """Returns a dictionary representation of the aspect."""
        return {
            "name": self.name,
            **({"description": self.description} if self.description else {}),
            **(({"children_keys": self.children_keys}) if self.children_keys else {}),
            **self.metadata,  # Include any additional metadata as extra fields
        }

    @property
    def context_dict(self):
        """Returns a dictionary composed of name, children_keys, root_elements, description and metadata."""
        return {
            "name": self.name,
            "children_keys": self.children_keys,
            "root_elements": self.root_element_names,
            **({"description": self.description} if self.description else {}),
            **self.metadata,  # Include any additional metadata as extra fields
        }
