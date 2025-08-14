"""LLM-based content proposer using Sequential Diverse Prompting."""

import json
import random
from typing import Any

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .proposals import ContentProposal, ContentProposer
from .tool import LLMToolMixin


class LLMContentProposer(LLMToolMixin, ContentProposer):
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

    async def propose(self, believes: list[str], step_description: str, context: list[str]) -> list[ContentProposal]:
        """Propose content based on current beliefs using Sequential Diverse Prompting.

        Args:
            believes: Current story beliefs
            step_description: Description of the current step
            context: Available story context

        Returns:
            List of content proposals with reasoning
        """
        # Stage 1: Generate diverse, detailed, non-conflicting perspectives
        perspectives = await self._generate_perspectives(believes, step_description, context)

        if not perspectives:
            # Fallback: Generate default proposal if perspective generation fails
            return await self._generate_fallback_proposal(believes, step_description, context)

        # Stage 2: Generate content for each perspective
        proposals = []
        for perspective in perspectives:
            content_proposal = await self._generate_content_from_perspective(
                perspective, believes, step_description, context
            )
            if content_proposal:
                proposals.append(content_proposal)

        # Ensure we have at least one proposal
        if not proposals:
            return await self._generate_fallback_proposal(believes, step_description, context)

        return proposals

    async def _generate_perspectives(self, believes: list[str], step_description: str, context: list[str]) -> list[dict[str, Any]]:
        """Generate diverse perspectives for content creation.

        Returns:
            List of perspective dictionaries with id, description, and rationale
        """
        response = await self.call_template(
            "generate_content_perspectives.jinja2",
            num_perspectives=self.num_proposals,
            step_description=step_description,
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
        step_description: str,
        context: list[str]
    ) -> ContentProposal | None:
        """Generate content based on a specific perspective.

        Args:
            perspective: Perspective dictionary with id, description, and rationale
            believes: Current story beliefs
            step_description: Description of the current step
            context: Available story context

        Returns:
            ContentProposal or None if generation fails
        """
        response = await self.call_template(
            "generate_content_from_perspective.jinja2",
            perspective_id=perspective["id"],
            perspective_description=perspective["description"],
            perspective_rationale=perspective["rationale"],
            step_description=step_description,
            context=context,
            believes=believes,
            json_format=True
        )

        # Parse the JSON response
        result = json.loads(response)
        content = result.get("content", "").strip()
        execution_notes = result.get("execution_notes", "")

        if content and len(content) > 10:  # Basic content validation
            # Create comprehensive reason from perspective info and execution notes
            reason = f"Perspective: {perspective['description']}. {execution_notes}"
            return ContentProposal(content=content, reason=reason)

        return None

    async def _generate_fallback_proposal(
        self,
        believes: list[str],
        step_description: str,
        context: list[str]
    ) -> list[ContentProposal]:
        """Generate a fallback proposal when perspective generation fails.

        Args:
            believes: Current story beliefs
            step_description: Description of the current step
            context: Available story context

        Returns:
            List containing a single fallback content proposal
        """
        # Create a simple direct prompt for content generation
        fallback_prompt = f"""
        Based on the following step description and context, generate appropriate story content:
        
        Step: {step_description}
        
        Context: {' '.join(context) if context else 'No specific context provided'}
        
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
                reason="Generated using fallback approach due to perspective generation failure"
            )]

        # Ultimate fallback
        return [ContentProposal(
            content="[Content generation failed - please provide more specific context or try again]",
            reason="Unable to generate content due to technical issues"
        )]
