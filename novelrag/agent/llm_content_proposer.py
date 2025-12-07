"""LLM-based content proposer using Sequential Diverse Prompting."""

import json
import logging
from typing import Any

from novelrag.llm import LLMMixin
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .proposals import ContentProposal, ContentProposer

logger = logging.getLogger(__name__)


class LLMContentProposer(LLMMixin, ContentProposer):
    """LLM-based content proposer using Sequential Diverse Prompting with dynamic perspective generation."""

    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM, num_proposals: int = 3):
        """Initialize the LLM content proposer.

        Args:
            template_env: Template environment for rendering prompts
            chat_llm: Chat LLM for generating content
            num_proposals: Number of content proposals to generate (default: 3)
        """
        super().__init__(template_env=template_env, chat_llm=chat_llm)
        self.num_proposals = num_proposals

    async def propose(self, believes: list[str], content_description: str, context: dict[str, list[str]]) -> list[ContentProposal]:
        """Propose content based on current beliefs using Sequential Diverse Prompting.

        Args:
            believes: Current story beliefs
            content_description: Specific description of what content to generate
            context: Available story context organized by facets

        Returns:
            List of content proposals with reasoning
        """        
        perspectives = await self._generate_perspectives(believes, content_description, context)
        logger.info(f"Generated {len(perspectives)} perspectives: {perspectives}")

        if not perspectives:
            return await self._generate_fallback_proposal(believes, content_description, context)

        proposals = []
        for perspective in perspectives:
            content_proposal = await self._generate_content_from_perspective(
                perspective, believes, content_description, context
            )
            if content_proposal:
                logger.debug(f"Generate Proposal from perspective {perspective['description']}: {content_proposal.content}")
                proposals.append(content_proposal)

        if not proposals:
            return await self._generate_fallback_proposal(believes, content_description, context)

        return proposals

    async def _generate_perspectives(self, believes: list[str], content_description: str, context: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Generate diverse perspectives for content creation.

        Returns:
            List of perspective dictionaries with id, description, and rationale
        """
        response = await self.call_template(
            "generate_content_perspectives.jinja2",
            num_perspectives=self.num_proposals,
            content_description=content_description,
            context=context,
            believes=believes,
            json_format=True
        )

        # Parse the JSON response
        result = json.loads(response)
        return result.get("perspectives", [])

    async def _generate_content_from_perspective(
        self,
        perspective: dict[str, Any],
        believes: list[str],
        content_description: str,
        context: dict[str, list[str]]
    ) -> ContentProposal | None:
        """Generate content based on a specific perspective.

        Args:
            perspective: Perspective dictionary with id, description, and rationale
            believes: Current story beliefs
            content_description: Specific description of what content to generate
            context: Available story context organized by facets

        Returns:
            ContentProposal or None if generation fails
        """
        response = await self.call_template(
            "generate_content_from_perspective.jinja2",
            perspective_description=perspective["description"],
            perspective_rationale=perspective["rationale"],
            content_description=content_description,
            context=context,
            believes=believes,
            json_format=False
        )

        # Response is now plain text content
        content = response.strip()

        if content and len(content) > 10:  # Basic content validation
            # Use only the perspective description for the reason
            reason = perspective['description']
            return ContentProposal(content=content, perspective=reason)

        return None

    async def _generate_fallback_proposal(
        self,
        believes: list[str],
        content_description: str,
        context: dict[str, list[str]]
    ) -> list[ContentProposal]:
        """Generate a fallback proposal when perspective generation fails.

        Args:
            believes: Current story beliefs
            content_description: Specific description of what content to generate
            context: Available story context organized by facets

        Returns:
            List containing a single fallback content proposal
        """
        # Create a simple direct prompt for content generation
        # Flatten context for fallback prompt
        context_text = ""
        if context:
            for facet, items in context.items():
                if items:
                    context_text += f"\n{facet}:\n" + "\n".join(f"- {item}" for item in items)
        
        fallback_prompt = f"""
        Based on the following content description and context, generate appropriate story content:
        
        Content Description: {content_description}
        
        Context: {context_text if context_text else 'No specific context provided'}
        
        Story Beliefs: {' '.join(believes) if believes else 'No current beliefs'}
        
        Generate coherent, engaging content that addresses the step requirements.
        """

        content = await self.chat_llm.chat(messages=[
            {'role': 'system', 'content': 'You are a creative writing assistant. Generate story content based on the provided requirements.'},
            {'role': 'user', 'content': fallback_prompt}
        ])

        if content.strip():
            return [ContentProposal(
                content=content.strip(),
                perspective="Generated using fallback approach due to perspective generation failure"
            )]

        # Ultimate fallback
        return [ContentProposal(
            content="[Content generation failed - please provide more specific context or try again]",
            perspective="Unable to generate content due to technical issues"
        )]
