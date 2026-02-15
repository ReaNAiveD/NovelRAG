"""LLM-based content proposer using Sequential Diverse Prompting."""

import logging
import random
from typing import Annotated

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm
from novelrag.resource_agent.workspace import ContextSnapshot

from .proposals import ContentProposal, ContentProposer

logger = logging.getLogger(__name__)


class Perspective(BaseModel):
    """A single creative perspective for content generation."""
    id: Annotated[str, Field(description="Short identifier for the perspective.")]
    description: Annotated[str, Field(description="Description of the creative angle.")]
    rationale: Annotated[str, Field(description="Why this perspective is relevant.")]


class PerspectivesResponse(BaseModel):
    """LLM response containing diverse perspectives."""
    perspectives: Annotated[list[Perspective], Field(
        default_factory=list,
        description="List of diverse perspectives for content generation.",
    )]


class LLMContentProposer(ContentProposer):
    """LLM-based content proposer using Sequential Diverse Prompting with dynamic perspective generation."""

    PACKAGE_NAME = "novelrag.resource_agent.propose"
    PERSPECTIVES_TEMPLATE = "generate_content_perspectives.jinja2"
    CONTENT_TEMPLATE = "generate_content_from_perspective.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en", num_perspectives: int = 5, num_proposals: int = 2):
        """Initialize the LLM content proposer.

        Args:
            chat_llm: Chat LLM for generating content
            lang: Language for prompt templates (default: "en")
            num_proposals: Number of content proposals to generate (default: 3)
        """
        self.chat_llm = chat_llm
        self._perspectives_llm = chat_llm.with_structured_output(PerspectivesResponse)
        self.num_perspectives = num_perspectives
        self.num_proposals = num_proposals
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._perspectives_tmpl = template_env.load_template(self.PERSPECTIVES_TEMPLATE)
        self._content_tmpl = template_env.load_template(self.CONTENT_TEMPLATE)

    async def propose(self, believes: list[str], content_description: str, context: ContextSnapshot) -> list[ContentProposal]:
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
        perspectives = random.sample(perspectives, min(self.num_proposals, len(perspectives)))
        proposals = []
        for perspective in perspectives:
            content_proposal = await self._generate_content_from_perspective(
                perspective, believes, content_description, context
            )
            if content_proposal:
                logger.debug(f"Generate Proposal from perspective {perspective.description}: {content_proposal.content}")
                proposals.append(content_proposal)

        if not proposals:
            return await self._generate_fallback_proposal(believes, content_description, context)

        return proposals

    @trace_llm("content_perspectives")
    async def _generate_perspectives(self, believes: list[str], content_description: str, context: ContextSnapshot) -> list[Perspective]:
        """Generate diverse perspectives for content creation.

        Returns:
            List of Perspective objects with id, description, and rationale
        """
        prompt = self._perspectives_tmpl.render(
            num_perspectives=self.num_perspectives,
            content_description=content_description,
            context=context,
            believes=believes,
        )
        response = await self._perspectives_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate diverse perspectives."),
        ])
        assert isinstance(response, PerspectivesResponse)
        return response.perspectives

    @trace_llm("content_generation")
    async def _generate_content_from_perspective(
        self,
        perspective: Perspective,
        believes: list[str],
        content_description: str,
        context: ContextSnapshot
    ) -> ContentProposal | None:
        """Generate content based on a specific perspective.

        Args:
            perspective: Perspective object with id, description, and rationale
            believes: Current story beliefs
            content_description: Specific description of what content to generate
            context: Context snapshot from the resource workspace

        Returns:
            ContentProposal or None if generation fails
        """
        prompt = self._content_tmpl.render(
            perspective_description=perspective.description,
            perspective_rationale=perspective.rationale,
            content_description=content_description,
            context=context,
            believes=believes,
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate content based on the given perspective."),
        ])
        assert isinstance(response.content, str), "Expected string content from LLM response"

        content = response.content.strip()

        if content and len(content) > 10:  # Basic content validation
            # Use only the perspective description for the reason
            reason = perspective.description
            return ContentProposal(content=content, perspective=reason)

        return None

    @trace_llm("content_fallback")
    async def _generate_fallback_proposal(
        self,
        believes: list[str],
        content_description: str,
        context: ContextSnapshot
    ) -> list[ContentProposal]:
        """Generate a fallback proposal when perspective generation fails.

        Args:
            believes: Current story beliefs
            content_description: Specific description of what content to generate
            context: Context snapshot from the resource workspace

        Returns:
            List containing a single fallback content proposal
        """
        # Create a simple direct prompt for content generation
        # Flatten context for fallback prompt
        context_text = ""
        if context and context.segments:
            for segment in context.segments:
                if segment.included_data or segment.relations:
                    context_text += f"\n{segment.uri}:\n"
                    context_text += "\n".join(f"- {k}: {v}" for k, v in segment.included_data.items())
                    context_text += "\n".join(f"- Related to {uri}: {desc}" for uri, desc in segment.relations.items())
        
        fallback_prompt = f"""
        Based on the following content description and context, generate appropriate story content:
        
        Content Description: {content_description}
        
        Context: {context_text if context_text else 'No specific context provided'}
        
        Story Beliefs: {' '.join(believes) if believes else 'No current beliefs'}
        
        Generate coherent, engaging content that addresses the step requirements.
        """

        result = await self.chat_llm.ainvoke([
            SystemMessage(content='You are a creative writing assistant. Generate story content based on the provided requirements.'),
            HumanMessage(content=fallback_prompt),
        ])
        content = result.content if isinstance(result.content, str) else str(result.content)

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
