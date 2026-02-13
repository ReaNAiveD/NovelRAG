import json
import logging

from novelrag.agenturn.goal import Goal, AutonomousSource, GoalDecider
from novelrag.llm.mixin import LLMMixin
from langchain_core.language_models import BaseChatModel
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class BacklogGoalDecider(LLMMixin):
    """Generates goals from the backlog of pending work items.

    Reads the highest-priority backlog entries and uses the LLM to
    formulate a concrete, actionable goal. Pops the selected entry
    after goal creation.
    """

    TEMPLATE_NAME = "goal_from_backlog.jinja2"
    TOP_N = 5

    def __init__(self, backlog: Backlog[BacklogEntry], chat_llm: BaseChatModel, template_env: TemplateEnvironment):
        LLMMixin.__init__(self, template_env=template_env, chat_llm=chat_llm)
        self.backlog = backlog

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

        response = await self.call_template(
            template_name=self.TEMPLATE_NAME,
            json_format=True,
            backlog_entries=entry_summaries,
            beliefs=beliefs,
        )

        try:
            result = json.loads(response)
            goal_description = result.get("goal", "").strip()
            # LLM reports 1-based entry numbers it selected
            selected = result.get("selected_entries", [1])
        except (json.JSONDecodeError, AttributeError):
            goal_description = response.strip()
            selected = [1]

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
