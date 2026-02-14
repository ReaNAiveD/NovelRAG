import logging
from typing import Annotated

from pydantic import BaseModel, Field

from novelrag.agenturn.goal import Goal, AutonomousSource, GoalDecider
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class BacklogGoalResponse(BaseModel):
    """LLM response for backlog-based goal generation."""
    goal: Annotated[str, Field(description="A clear, actionable goal statement derived from backlog entries.")]
    selected_entries: Annotated[list[int], Field(
        default=[1],
        description="1-based indices of the backlog entries selected for this goal.",
    )]


class BacklogGoalDecider:
    """Generates goals from the backlog of pending work items.

    Reads the highest-priority backlog entries and uses the LLM to
    formulate a concrete, actionable goal. Pops the selected entry
    after goal creation.
    """

    PACKAGE_NAME = "novelrag.resource_agent.goal_decider"
    TEMPLATE_NAME = "goal_from_backlog.jinja2"
    TOP_N = 5

    def __init__(self, backlog: Backlog[BacklogEntry], chat_llm: BaseChatModel, lang: str = "en"):
        self._goal_llm = chat_llm.with_structured_output(BacklogGoalResponse)
        self.backlog = backlog
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._template = template_env.load_template(self.TEMPLATE_NAME)

    async def next_goal(self, beliefs: list[str]) -> Goal | None:
        if len(self.backlog) == 0:
            logger.debug("BacklogGoalDecider: backlog is empty, skipping.")
            return None

        top_entries = self.backlog.get_top(self.TOP_N)
        entry_summaries = [
            {
                "type": entry.type,
                "priority": entry.priority,
                "description": entry.description,
                **entry.metadata,
            }
            for entry in top_entries
        ]

        prompt = self._template.render(
            backlog_entries=entry_summaries,
            beliefs=beliefs,
        )
        response = await self._goal_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate a goal from the backlog."),
        ])
        assert isinstance(response, BacklogGoalResponse)

        goal_description = response.goal.strip()
        selected = response.selected_entries

        if not goal_description:
            return None

        # Convert 1-based selected_entries to 0-based indices into top_entries,
        # then map to positions in the full backlog (which shares the same order).
        indices_to_remove = [
            i - 1 for i in selected
            if isinstance(i, int) and 1 <= i <= len(top_entries)
        ]
        if not indices_to_remove:
            # Fallback: remove the first entry if LLM didn't provide valid indices
            indices_to_remove = [0]

        removed = self.backlog.remove_entries(indices_to_remove)
        context_parts = [
            f"(priority={e.priority}) {e.description[:80]}" for e in removed
        ]

        return Goal(
            description=goal_description,
            source=AutonomousSource(
                decider_name="backlog",
                context=f"Based on backlog entries: {'; '.join(context_parts)}",
            ),
        )
