from novelrag.agenturn.pursuit import LLMPursuitAssessor
from novelrag.agenturn.types import InteractionContext
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import UndoQueue
from novelrag.utils.language import content_directive

from .goal_decider import CompositeGoalDecider
from .action_determine import (
    ActionDetermineLoop,
    ActionDecider,
    LLMActionDecider,
    ContextAnalyser,
    LLMContextAnalyzer,
    ContextDiscoverer,
    LLMContextDiscoverer,
    RefinementAnalyzer,
    LLMRefinementAnalyzer,
)
from .workspace import (
    ResourceContext,
    ContextWorkspace,
    ContextSnapshot,
    ResourceSegment,
    SegmentData,
    SearchHistoryItem,
)
from .propose import ContentProposal, ContentProposer, LLMContentProposer

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
    "ActionDecider",
    "ContextAnalyser",
    "ContextDiscoverer",
    "RefinementAnalyzer",
    "LLMContextDiscoverer",
    "LLMContextAnalyzer",
    "LLMActionDecider",
    "LLMRefinementAnalyzer",
    
    # Workspace
    "ResourceContext",
    "ContextWorkspace",
    "ContextSnapshot",
    "ResourceSegment",
    "SegmentData",
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
        lang: Content language code (e.g. "zh", "en"). When None, language
              is inferred from beliefs by the LLM.
    
    Returns:
        GoalExecutor configured with resource tools and OrchestrationLoop
    """
    from novelrag.agenturn import GoalExecutor

    # Build language directive once and pass to all components
    lang_directive = content_directive(lang, beliefs)

    # Create workspace and orchestrator
    context = ResourceContext(resource_repo)
    pursuit_assessor = LLMPursuitAssessor(chat_llm=chat_llm, lang=lang or "en", lang_directive=lang_directive)
    discoverer = LLMContextDiscoverer(chat_llm, lang=lang or "en", lang_directive=lang_directive)
    analyser = LLMContextAnalyzer(chat_llm, lang=lang or "en", lang_directive=lang_directive)
    decider = LLMActionDecider(chat_llm, lang=lang or "en", lang_directive=lang_directive)
    refiner = LLMRefinementAnalyzer(chat_llm, lang=lang or "en", lang_directive=lang_directive)
    orchestrator = ActionDetermineLoop(
        context=context,
        pursuit_assessor=pursuit_assessor,
        discoverer=discoverer,
        analyser=analyser,
        decider=decider,
        refiner=refiner,
    )

    # Create resource tools
    tools = {
        "AspectCreateTool": AspectCreateTool(resource_repo, chat_llm, lang=lang or "en", lang_directive=lang_directive),
        "ResourceWriteTool": ResourceWriteTool(resource_repo, context, chat_llm, lang=lang or "en", lang_directive=lang_directive, backlog=backlog, undo_queue=undo_queue),
        "ResourceRelationWriteTool": ResourceRelationWriteTool(resource_repo, chat_llm, lang=lang or "en", lang_directive=lang_directive),
    }

    return GoalExecutor(
        beliefs=beliefs or [],
        tools=tools,
        determiner=orchestrator,
        channel=channel,
    )
