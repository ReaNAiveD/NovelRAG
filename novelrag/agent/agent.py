"""Main Agent class for orchestrating execution."""

import logging
from typing import Any

from novelrag.agent.channel import AgentChannel
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .tool import BaseTool, ContextualTool, LLMToolMixin, SchematicTool
from .schedule import GoalPlanner, GoalPursuit, GoalPursuitResult
from .proposals import TargetProposer

logger = logging.getLogger(__name__)


class AgentMind:
    """The cognitive core of the agent - beliefs, goals, and decision-making.
    
    This represents the agent's mental model of the world, its objectives,
    and its ability to reason and make decisions.
    """
    
    def __init__(self):
        self.beliefs: list[str] = []  # What the agent believes to be true
        self.current_goal: str | None = None  # What the agent is trying to achieve
        self.backlog: list[str] = []  # Things the agent wants to remember for later
        
    def set_goal(self, goal: str):
        """Set a new primary goal for the agent to pursue."""
        self.current_goal = goal
    
    def add_belief(self, belief: str):
        """Add a new belief to the agent's world model."""
        if belief not in self.beliefs:
            self.beliefs.append(belief)
    
    def remove_belief(self, belief: str):
        """Remove a belief that is no longer valid."""
        if belief in self.beliefs:
            self.beliefs.remove(belief)
    
    def add_to_backlog(self, item: str):
        """Remember something for later consideration."""
        if item not in self.backlog:
            self.backlog.append(item)
    
    def get_world_view(self) -> dict[str, Any]:
        """Get the agent's current understanding of the world."""
        return {
            "beliefs": self.beliefs.copy(),
            "current_goal": self.current_goal,
            "backlog_items": len(self.backlog)
        }


class Agent(LLMToolMixin):
    """Main agent class for orchestrating tool execution and managing state.
    
    This is a composition of the core agent components:
    - AgentMind: Handles beliefs, goals, and mental state
    - AgentCommunicator: Manages all communication and output
    - AgentExecutor: Executes actions and tools
    """

    def __init__(self, tools: dict[str, BaseTool], channel: AgentChannel, planner: GoalPlanner, template_env: TemplateEnvironment, chat_llm: ChatLLM):
        # Initialize core components
        self.mind = AgentMind()
        self.channel = channel

        self.planner = planner
        self.tools: dict[str, BaseTool] = tools
        self.contextual_tools: dict[str, ContextualTool] = dict()
        for (name, tool) in self.tools.items():
            if isinstance(tool, ContextualTool):
                self.contextual_tools[name] = tool
            elif isinstance(tool, SchematicTool):
                self.contextual_tools[name] = tool.wrapped(template_env, chat_llm)
            else:
                logger.warning(f'Tool {name} is not a ContextualTool or SchematicTool, skipping tool setup.')
        self.target_proposers: list[TargetProposer] = []
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    @property
    def believes(self) -> list[str]:
        """Access beliefs through the mind component."""
        return self.mind.beliefs
    
    @property
    def target(self) -> str | None:
        """Access current goal through the mind component."""
        return self.mind.current_goal
    
    @property
    def backlog(self) -> list[str]:
        """Access backlog through the mind component."""
        return self.mind.backlog

    def decide(self):
        """Auto Decision - delegates to mind component."""
        # TODO: Implement decision logic in mind component
        pass

    async def handle_goal(self, goal: str):
        pursuit = await self.planner.plan_goal(goal, self.believes, self.contextual_tools)
        while isinstance(pursuit, GoalPursuit):
            pursuit = await pursuit.advance(self.contextual_tools, self.channel)
        if isinstance(pursuit, GoalPursuitResult):
            records = pursuit.records.completed_steps
            if not records:
                await self.channel.error(f'No steps completed for goal "{goal}".')
                return
            last_step = records[-1]
            for result in last_step.results:
                await self.channel.output(result)
