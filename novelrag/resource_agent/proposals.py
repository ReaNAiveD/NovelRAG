"""Content proposal classes for the resource agent system."""


class ContentProposal:
    """Represents a proposed content change with reasoning."""
    
    def __init__(self, content: str, perspective: str):
        self.content = content
        self.reason = perspective


class ContentProposer:
    """Proposes content changes based on current beliefs and context."""
    
    async def propose(self, believes: list[str], content_description: str, context: dict[str, list[str]]) -> list[ContentProposal]:
        """Propose content based on current beliefs and targeted context.
        
        Args:
            believes: Current agent beliefs
            content_description: Specific description of what content to generate
            context: Full context dictionary
            
        Returns:
            List of content proposals with reasoning
        """
        raise NotImplementedError("Subclasses should implement this method.")
