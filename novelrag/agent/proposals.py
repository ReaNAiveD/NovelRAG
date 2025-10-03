"""Proposal classes for target and content generation."""


class ProposalSelector:
    """Selects proposals from a list of options."""
    
    async def select(self, proposals: list[str]) -> str:
        """Select a proposal from the list."""
        raise NotImplementedError("TODO: Implement the logic to select a proposal.")


class TargetProposal:
    """Represents a proposed target with reasoning."""
    
    def __init__(self, target: str, reason: str):
        self.target = target
        self.reason = reason


class TargetProposer:
    """Proposes targets based on current beliefs."""
    
    async def propose(self, believes: list[str]) -> list[TargetProposal]:
        """Propose targets based on current beliefs."""
        raise NotImplementedError("Subclasses should implement this method.")


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
