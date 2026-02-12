"""Unified exploration goal decider.

Replaces the former RandomWalkGoalDecider and RandomAspectGoalDecider with a
single decider that handles three tiers of repository state:

* **Full cold start** – no aspects exist; bootstrap from beliefs alone.
* **Partial cold start** – aspects exist but contain no elements; populate them.
* **Normal operation** – elements exist; random-walk to one, expand context,
  analyse concept gaps, and produce a focused goal.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from novelrag.agenturn.goal import Goal, AutonomousSource
from novelrag.llm.mixin import LLMMixin
from novelrag.llm.types import ChatLLM
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.goal_decider.recency import RecencyWeighter
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context bundle produced by the expansion step
# ---------------------------------------------------------------------------

@dataclass
class _ResolvedReference:
    """A relationship target that exists in the repository."""
    uri: str
    id: str
    aspect: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _UnresolvedReference:
    """A relationship target declared on an element but absent from the repo."""
    uri: str
    declared_descriptions: list[str] = field(default_factory=list)


@dataclass
class _AspectSummary:
    name: str
    description: str | None
    element_count: int
    sample_element_ids: list[str]


@dataclass
class _ContextBundle:
    """Aggregated context gathered around a selected element."""
    related_resources: list[dict[str, Any]]
    resolved_refs: list[_ResolvedReference]
    unresolved_refs: list[_UnresolvedReference]
    aspect_summaries: list[_AspectSummary]


class ExplorationGoalDecider(LLMMixin):
    """Generates autonomous goals by exploring the resource repository.

    Three operational branches based on repository state:

    * *Bootstrap* – no aspects → proposes first aspect(s) from beliefs.
    * *Populate* – aspects present, no elements → populates an aspect.
    * *Explore* – elements present → random-walk to one, expand context,
      run concept-gap analysis, then generate and refine a goal.
    """

    BOOTSTRAP_TEMPLATE = "bootstrap_from_beliefs.jinja2"
    CONTEXT_DISCOVERY_TEMPLATE = "exploration_context_discovery.jinja2"
    GAP_ANALYSIS_TEMPLATE = "concept_gap_analysis.jinja2"
    GOAL_TEMPLATE = "goal_from_exploration.jinja2"
    SEARCH_LIMIT = 5

    def __init__(
        self,
        repo: ResourceRepository,
        chat_llm: ChatLLM,
        template_env: TemplateEnvironment,
        recency: RecencyWeighter | None = None,
    ):
        LLMMixin.__init__(self, template_env=template_env, chat_llm=chat_llm)
        self.repo = repo
        self.recency = recency

    async def next_goal(self, beliefs: list[str]) -> Goal | None:
        aspects = await self.repo.all_aspects()

        if not aspects:
            return await self._bootstrap(beliefs)

        # Collect every element across all aspects
        all_elements: list[tuple[ResourceAspect, DirectiveElement]] = []
        for aspect in aspects:
            for element in aspect.iter_elements():
                all_elements.append((aspect, element))

        if not all_elements:
            return await self._populate(aspects, beliefs)

        return await self._explore(aspects, all_elements, beliefs)

    async def _bootstrap(self, beliefs: list[str]) -> Goal | None:
        logger.info("ExplorationGoalDecider: no aspects – bootstrapping from beliefs.")

        response = await self.call_template(
            template_name=self.BOOTSTRAP_TEMPLATE,
            json_format=True,
            beliefs=beliefs,
        )

        try:
            result = json.loads(response)
            goal_description = result.get("goal", "").strip()
        except (json.JSONDecodeError, AttributeError):
            goal_description = response.strip()

        if not goal_description:
            return None

        return Goal(
            description=goal_description,
            source=AutonomousSource(
                decider_name="exploration",
                context="phase=bootstrap",
            ),
        )

    async def _populate(
        self,
        aspects: list[ResourceAspect],
        beliefs: list[str],
    ) -> Goal | None:
        logger.info("ExplorationGoalDecider: aspects exist but no elements – populating.")

        # Pick an aspect (recency-weighted if available)
        if self.recency is not None:
            weights = self.recency.aspect_weights([a.name for a in aspects])
            aspect = random.choices(aspects, weights=weights, k=1)[0]
        else:
            aspect = random.choice(aspects)

        response = await self.call_template(
            template_name=self.GOAL_TEMPLATE,
            json_format=True,
            element=None,
            gap_analysis=None,
            focus="populate",
            aspect=aspect.aspect_dict,
            aspect_summaries=[
                {"name": a.name, "description": a.description}
                for a in aspects
            ],
            beliefs=beliefs,
        )

        try:
            result = json.loads(response)
            goal_description = result.get("goal", "").strip()
        except (json.JSONDecodeError, AttributeError):
            goal_description = response.strip()

        if not goal_description:
            return None

        return Goal(
            description=goal_description,
            source=AutonomousSource(
                decider_name="exploration",
                context=f"phase=populate, aspect='{aspect.name}'",
            ),
        )

    async def _explore(
        self,
        aspects: list[ResourceAspect],
        all_elements: list[tuple[ResourceAspect, DirectiveElement]],
        beliefs: list[str],
    ) -> Goal | None:
        # 1. Select element via random walk (recency-biased)
        if self.recency is not None:
            weights = self.recency.element_weights(
                [(a.name, e.inner.uri) for a, e in all_elements]
            )
            aspect, element = random.choices(all_elements, weights=weights, k=1)[0]
        else:
            aspect, element = random.choice(all_elements)

        logger.info(
            "ExplorationGoalDecider: walked to '%s' in aspect '%s'.",
            element.inner.uri, aspect.name,
        )

        # 2. Context expansion (LLM-driven discovery + resolution)
        ctx = await self._expand_context(element, aspect, aspects, beliefs)

        # 3. Concept-gap analysis (LLM call #1)
        gap_analysis = await self._analyse_gaps(element, aspect, ctx, beliefs)

        focus = gap_analysis.get("priority_concern", "enrichment")
        if focus not in (
            "creation_aspect", "creation_element", "enrichment", "verification",
        ):
            focus = "enrichment"

        # 4. Generate goal (LLM call #3)
        element_content = {
            "uri": element.inner.uri,
            "id": element.id,
            "aspect": aspect.name,
            "properties": element.inner.props(),
            "relationships": element.inner.relationships,
        }

        response = await self.call_template(
            template_name=self.GOAL_TEMPLATE,
            json_format=True,
            element=element_content,
            gap_analysis=gap_analysis,
            focus=focus,
            aspect=aspect.aspect_dict,
            aspect_summaries=[
                {
                    "name": s.name,
                    "description": s.description,
                    "element_count": s.element_count,
                }
                for s in ctx.aspect_summaries
            ],
            beliefs=beliefs,
        )

        try:
            result = json.loads(response)
            goal_description = result.get("goal", "").strip()
        except (json.JSONDecodeError, AttributeError):
            goal_description = response.strip()

        if not goal_description:
            return None

        return Goal(
            description=goal_description,
            source=AutonomousSource(
                decider_name="exploration",
                context=(
                    f"phase=explore, element='{element.inner.uri}', "
                    f"aspect='{aspect.name}', focus={focus}"
                ),
            ),
        )

    async def _expand_context(
        self,
        element: DirectiveElement,
        aspect: ResourceAspect,
        aspects: list[ResourceAspect],
        beliefs: list[str],
    ) -> _ContextBundle:
        """Gather context around *element* using LLM-driven discovery.

        1. Build aspect summaries (code).
        2. Ask the LLM which URIs to load and what to search for.
        3. Resolve those URIs and run the suggested search queries.
        """

        # --- aspect summaries (code-only) ---
        summaries: list[_AspectSummary] = []
        for asp in aspects:
            elements_iter = list(asp.iter_elements())
            sample_ids = [e.id for e in elements_iter[:10]]
            summaries.append(
                _AspectSummary(
                    name=asp.name,
                    description=asp.description,
                    element_count=len(elements_iter),
                    sample_element_ids=sample_ids,
                )
            )

        # --- LLM-driven discovery ---
        element_content = {
            "uri": element.inner.uri,
            "id": element.id,
            "aspect": aspect.name,
            "properties": element.inner.props(),
            "relationships": element.inner.relationships,
            "children_ids": element.inner.flattened_child_ids(),
        }
        aspect_summary_dicts = [
            {
                "name": s.name,
                "description": s.description,
                "element_count": s.element_count,
                "sample_element_ids": s.sample_element_ids,
            }
            for s in summaries
        ]

        response = await self.call_template(
            template_name=self.CONTEXT_DISCOVERY_TEMPLATE,
            json_format=True,
            element=element_content,
            aspect_summaries=aspect_summary_dicts,
            beliefs=beliefs,
        )

        try:
            discovery = json.loads(response)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("ExplorationGoalDecider: context discovery returned non-JSON; falling back.")
            discovery = {}

        query_uris: list[str] = discovery.get("query_resources", [])
        search_queries: list[str] = discovery.get("search_queries", [])

        # Always include declared relationship URIs so we get
        # factual resolved/unresolved data for gap analysis.
        declared_uris = set(element.inner.relationships.keys())

        # Walk up the parent chain to include ancestor element / aspect URIs.
        ancestor_uris: set[str] = set()
        cur = element.parent
        while cur is not None:
            ancestor_uris.add(cur.inner.uri)
            cur = cur.parent
        # The aspect itself acts as the root container.
        ancestor_uris.add(f"/{aspect.name}")

        all_uris_to_resolve = list(
            declared_uris | set(query_uris) | ancestor_uris
        )

        # --- resolve URIs ---
        resolved: list[_ResolvedReference] = []
        unresolved: list[_UnresolvedReference] = []

        for uri in all_uris_to_resolve:
            # Skip the element itself
            if uri == element.inner.uri:
                continue
            found = await self.repo.find_by_uri(uri)
            descriptions = element.inner.relationships.get(uri, [])
            if found is None:
                unresolved.append(
                    _UnresolvedReference(uri=uri, declared_descriptions=descriptions)
                )
            elif isinstance(found, DirectiveElement):
                resolved.append(
                    _ResolvedReference(
                        uri=found.inner.uri,
                        id=found.id,
                        aspect=found.inner.aspect,
                        properties=found.inner.props(),
                    )
                )
            elif isinstance(found, ResourceAspect):
                resolved.append(
                    _ResolvedReference(
                        uri=f"/{found.name}", id=found.name, aspect=found.name
                    )
                )

        # --- run LLM-suggested search queries ---
        related_resources: list[dict[str, Any]] = []
        seen_uris: set[str] = {element.inner.uri}

        for query in search_queries:
            if not query or not query.strip():
                continue
            results = await self.repo.vector_search(
                query.strip(), limit=self.SEARCH_LIMIT
            )
            for sr in results:
                uri = sr.element.inner.uri
                if uri not in seen_uris:
                    seen_uris.add(uri)
                    related_resources.append(
                        {"uri": uri, "id": sr.element.id, "distance": sr.distance}
                    )

        return _ContextBundle(
            related_resources=related_resources,
            resolved_refs=resolved,
            unresolved_refs=unresolved,
            aspect_summaries=summaries,
        )

    async def _analyse_gaps(
        self,
        element: DirectiveElement,
        aspect: ResourceAspect,
        ctx: _ContextBundle,
        beliefs: list[str],
    ) -> dict[str, Any]:
        """Run concept-gap analysis over the expanded context (LLM call)."""

        element_content = {
            "uri": element.inner.uri,
            "id": element.id,
            "aspect": aspect.name,
            "properties": element.inner.props(),
            "relationships": element.inner.relationships,
        }

        resolved_refs = [
            {"uri": r.uri, "id": r.id, "aspect": r.aspect, "properties": r.properties}
            for r in ctx.resolved_refs
        ]
        unresolved_refs = [
            {"uri": u.uri, "declared_descriptions": u.declared_descriptions}
            for u in ctx.unresolved_refs
        ]
        aspect_summaries = [
            {
                "name": s.name,
                "description": s.description,
                "element_count": s.element_count,
                "sample_element_ids": s.sample_element_ids,
            }
            for s in ctx.aspect_summaries
        ]

        response = await self.call_template(
            template_name=self.GAP_ANALYSIS_TEMPLATE,
            json_format=True,
            element=element_content,
            related_resources=ctx.related_resources,
            resolved_refs=resolved_refs,
            unresolved_refs=unresolved_refs,
            aspect_summaries=aspect_summaries,
            beliefs=beliefs,
        )

        try:
            return json.loads(response)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("ExplorationGoalDecider: gap analysis returned non-JSON; defaulting to enrichment.")
            return {"priority_concern": "enrichment", "reasoning": response.strip()}
