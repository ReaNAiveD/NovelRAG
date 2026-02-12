from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import UndoQueue
from .goal_decider import CompositeGoalDecider
from .action_determine_loop import (
    ActionDetermineLoop,
    DiscoveryPlan,
    RefinementPlan,
    ActionDecision,
    RefinementDecision,
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
    
    # Goal decider
    "CompositeGoalDecider",
    
    # Factory
    "create_executor",
]


def create_executor(
    resource_repo,
    channel,
    chat_llm,
    beliefs: list[str] | None = None,
    lang: str | None = None,
    backlog: Backlog[BacklogEntry] | None = None,
    undo_queue: UndoQueue | None = None,
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
    orchestrator = ActionDetermineLoop(context, chat_llm, template_lang=lang or "en")
    
    # Create resource tools
    tools = {
        "AspectCreateTool": AspectCreateTool(resource_repo, resource_template_env, chat_llm),
        "ResourceWriteTool": ResourceWriteTool(resource_repo, context, resource_template_env, chat_llm, backlog=backlog, undo_queue=undo_queue),
        "ResourceRelationWriteTool": ResourceRelationWriteTool(resource_repo, resource_template_env, chat_llm),
    }
    
    return GoalExecutor(
        beliefs=beliefs or [],
        tools=tools,
        determiner=orchestrator,
        channel=channel,
    )
