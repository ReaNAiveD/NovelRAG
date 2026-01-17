"""Resource-specific agent module built on top of the agenturn framework.

This module provides:
- OrchestrationLoop: Context-driven multi-phase action determination
- ResourceContext/ContextWorkspace: Resource workspace management
- Resource tools: Fetch, search, write, aspect, relation operations
- Content proposers: LLM-based content generation

Factory function:
- create_resource_agent(): Creates a GoalExecutor configured for resource operations
"""

from .action_determine_loop import (
    ActionDetermineLoop,
    DiscoveryPlan,
    RefinementPlan,
    ActionDecision,
    RefinementDecision,
    OrchestrationExecutionPlan,
    OrchestrationFinalization,
)
from .workspace import (
    ResourceContext,
    ContextWorkspace,
    ResourceSegment,
    SearchHistoryItem,
)
from .proposals import ContentProposal, ContentProposer
from .llm_content_proposer import LLMContentProposer

from .tool import (
    ResourceFetchTool,
    ResourceSearchTool,
    ResourceWriteTool,
    AspectCreateTool,
    ResourceRelationWriteTool,
    ContentGenerationTask,
)

__all__ = [
    # Orchestration
    "ActionDetermineLoop",
    "DiscoveryPlan",
    "RefinementPlan",
    "ActionDecision",
    "RefinementDecision",
    "OrchestrationExecutionPlan",
    "OrchestrationFinalization",
    
    # Workspace
    "ResourceContext",
    "ContextWorkspace",
    "ResourceSegment",
    "SearchHistoryItem",
    
    # Content proposers
    "ContentProposal",
    "ContentProposer",
    "LLMContentProposer",
    
    # Resource tools
    "ResourceFetchTool",
    "ResourceSearchTool",
    "ResourceWriteTool",
    "AspectCreateTool",
    "ResourceRelationWriteTool",
    "ContentGenerationTask",
    
    # Factory
    "create_executor",
]


def create_executor(
    resource_repo,
    channel,
    chat_llm,
    beliefs: list[str] | None = None,
    lang: str = "en",
):
    """Factory function to create a GoalExecutor configured for resource operations.
    
    Args:
        resource_repo: ResourceRepository instance for data access
        channel: AgentChannel for communication
        chat_llm: ChatLLM instance for LLM calls
        beliefs: Optional list of agent beliefs/constraints
        lang: Language for templates (default: "en")
    
    Returns:
        GoalExecutor configured with resource tools and OrchestrationLoop
    """
    from novelrag.agenturn import GoalExecutor
    from novelrag.template import TemplateEnvironment

    resource_template_env = TemplateEnvironment("novelrag.resource_agent", lang)
    
    # Create workspace and orchestrator
    context = ResourceContext(resource_repo, resource_template_env, chat_llm)
    orchestrator = ActionDetermineLoop(context, chat_llm, template_lang=lang)
    
    # Create resource tools
    tools = {
        "ResourceFetchTool": ResourceFetchTool(resource_repo),
        "ResourceSearchTool": ResourceSearchTool(resource_repo),
        "AspectCreateTool": AspectCreateTool(resource_repo, resource_template_env, chat_llm),
        "ResourceRelationWriteTool": ResourceRelationWriteTool(resource_repo, resource_template_env, chat_llm),
    }
    
    return GoalExecutor(
        beliefs=beliefs or [],
        tools=tools,
        determiner=orchestrator,
        channel=channel,
    )
