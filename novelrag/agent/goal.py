"""Goal building utilities for translating user requests into goals."""

from novelrag.llm import LLMMixin


class GoalBuilder(LLMMixin):
    """Builds clear and concise goals from user requests using LLM."""

    async def build_goal(self, user_request: str) -> str:
        """Build a clear and concise goal from the user's request."""
        goal = await self.call_template(
            "translate_request_to_goal.jinja2",
            user_request=user_request
        )
        if goal.startswith("**Goal**: "):
            goal = goal[len("**Goal**: "):]
        return goal.strip()
