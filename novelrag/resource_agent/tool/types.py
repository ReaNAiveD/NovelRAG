"""Type definitions for resource agent tools."""

from dataclasses import dataclass, field


@dataclass
class ContentGenerationTask:
    """Represents a content generation task."""
    description: str                    # What content to generate
    content_key: str | None = None    # Which field this content will update
