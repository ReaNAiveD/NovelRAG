"""Type definitions for write tools."""

from dataclasses import dataclass, field


@dataclass
class ContentGenerationTask:
    """Represents a content generation task."""
    description: str                    # What content to generate
    content_key: str | None = None    # Which field this content will update


@dataclass
class OperationPlan:
    """Result of operation planning phase."""
    operation_specification: str       # Detailed natural language intent
    content_generation_tasks: list[ContentGenerationTask]
    missing_aspects: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)

    @property
    def has_prerequisites(self) -> bool:
        """Check if there are any unmet prerequisites."""
        return bool(self.missing_aspects or self.missing_context)

    def prerequisites_error(self) -> str:
        """Generate error message for unmet prerequisites."""
        errors = []
        if self.missing_aspects:
            errors.append(f"Missing aspects: {', '.join(self.missing_aspects)}")
        if self.missing_context:
            errors.append(f"Missing context: {', '.join(self.missing_context)}")
        return "; ".join(errors)
