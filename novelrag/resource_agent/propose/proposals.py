from typing import Protocol

from novelrag.resource_agent.workspace import ContextSnapshot


class ContentProposal:
    """Represents a proposed content change with reasoning."""
    
    def __init__(self, content: str, perspective: str):
        self.content = content
        self.reason = perspective


class ContentProposer(Protocol):
    """Proposes content changes based on current beliefs and context."""
    
    async def propose(self, believes: list[str], content_description: str, context: ContextSnapshot) -> list[ContentProposal]:
        """Propose content based on current beliefs and targeted context.

        Args:
            believes: Current agent beliefs
            content_description: Specific description of what content to generate
            context: Context snapshot from the resource workspace

        Returns:
            List of content proposals with reasoning
        """
        ...
