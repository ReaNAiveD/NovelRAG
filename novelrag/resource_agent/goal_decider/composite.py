import logging
import random

from novelrag.agenturn.goal import Goal, GoalDecider
from langchain_core.language_models import BaseChatModel
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import UndoQueue

from .backlog_source import BacklogGoalDecider
from .exploration import ExplorationGoalDecider
from .recency import RecencyWeighter

logger = logging.getLogger(__name__)

# Default weights for each source decider
DEFAULT_WEIGHTS: dict[str, float] = {
    "backlog": 2.0,
    "exploration": 2.0,
}


class CompositeGoalDecider:
    """Hardcoded composite that manages all source deciders and selects among them.

    Constructs all inner GoalDecider instances internally and uses a fixed
    weighted-random selection strategy. Weights are dynamically adjusted
    based on data availability (e.g., backlog weight zeroed if backlog is empty).

    When an UndoQueue is provided, a RecencyWeighter is created and passed to
    the exploration decider so that recently-operated resources are
    down-weighted, promoting diversity.
    """

    def __init__(
        self,
        repo: ResourceRepository,
        chat_llm: BaseChatModel,
        template_lang: str = "en",
        backlog: Backlog[BacklogEntry] | None = None,
        undo_queue: UndoQueue | None = None,
        weight_overrides: dict[str, float] | None = None,
        lang_directive: str = "",
    ):
        self.repo = repo
        self.backlog = backlog
        self.undo_queue = undo_queue
        self.base_weights = {**DEFAULT_WEIGHTS, **(weight_overrides or {})}

        # Build recency weighter from undo queue (used by exploration deciders)
        recency = RecencyWeighter(undo_queue) if undo_queue is not None else None

        # Build inner deciders
        self._deciders: dict[str, GoalDecider] = {}

        if backlog is not None:
            self._deciders["backlog"] = BacklogGoalDecider(backlog, chat_llm, lang=template_lang, lang_directive=lang_directive)

        self._deciders["exploration"] = ExplorationGoalDecider(
            repo, chat_llm, lang=template_lang, recency=recency, lang_directive=lang_directive
        )

    async def next_goal(self, beliefs: list[str]) -> Goal | None:
        """Select a source decider via weighted random and delegate goal generation.

        If the selected decider returns None, it is excluded and another is tried,
        until all deciders are exhausted.
        """
        candidates = self._build_weighted_candidates()
        if not candidates:
            logger.info("CompositeGoalDecider: no candidate deciders available.")
            return None

        # Try deciders until one produces a goal or all are exhausted
        remaining = list(candidates)
        while remaining:
            names, weights = zip(*remaining)
            selected_name = random.choices(names, weights=weights, k=1)[0]

            logger.info(f"CompositeGoalDecider: selected source '{selected_name}'")
            decider = self._deciders[selected_name]

            goal = await decider.next_goal(beliefs)
            if goal is not None:
                return goal

            logger.debug(f"CompositeGoalDecider: '{selected_name}' returned None, trying another.")
            remaining = [(n, w) for n, w in remaining if n != selected_name]

        logger.info("CompositeGoalDecider: all deciders exhausted, no goal produced.")
        return None

    def _build_weighted_candidates(self) -> list[tuple[str, float]]:
        """Build the list of (name, weight) pairs, applying dynamic adjustments."""
        candidates = []

        for name, decider in self._deciders.items():
            weight = self.base_weights.get(name, 1.0)
            if weight <= 0:
                continue

            # Dynamic weight adjustments based on data availability
            adjusted_weight = self._adjust_weight(name, weight)
            if adjusted_weight > 0:
                candidates.append((name, adjusted_weight))

        return candidates

    def _adjust_weight(self, name: str, base_weight: float) -> float:
        """Apply dynamic weight adjustments based on current state."""
        if name == "backlog" and self.backlog is not None:
            if len(self.backlog) == 0:
                return 0.0
            # Boost weight proportionally to backlog size (capped)
            return base_weight * min(len(self.backlog) / 3.0, 2.0)

        return base_weight
