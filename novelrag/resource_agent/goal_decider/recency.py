"""Recency-based weight adjustment for goal deciders.

Reads recent operations from the UndoQueue and computes down-weight
factors so that recently-operated aspects/elements are less likely
to be chosen, promoting exploration diversity.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Sequence

from novelrag.resource_agent.undo import UndoQueue

logger = logging.getLogger(__name__)


class RecencyWeighter:
    """Derives per-aspect and per-element weight penalties from recent undo actions.

    Parameters
    ----------
    undo_queue:
        The undo queue to peek recent actions from.
    peek_count:
        How many recent actions to consider.
    decay:
        Controls how aggressively recent operations are penalised.
        Weight formula: ``1 / (1 + count * decay)``.
        Higher decay → stronger penalty per operation.
    """

    def __init__(
        self,
        undo_queue: UndoQueue,
        peek_count: int = 10,
        decay: float = 0.5,
    ) -> None:
        self.undo_queue = undo_queue
        self.peek_count = peek_count
        self.decay = decay

    def aspect_weights(self, aspect_names: Sequence[str]) -> list[float]:
        """Return a weight for each aspect name.

        Aspects touched more frequently in recent operations receive
        a lower (but always positive) weight.
        """
        if not aspect_names:
            return []

        counts = self._aspect_counts()
        if not counts:
            return [1.0] * len(aspect_names)

        return [self._decay_weight(counts.get(name, 0)) for name in aspect_names]

    def element_weights(
        self,
        elements: Sequence[tuple[str, str]],
        *,
        aspect_blend: float = 0.3,
    ) -> list[float]:
        """Return a weight for each ``(aspect_name, element_uri)`` pair.

        The weight blends element-level and aspect-level penalties so
        that the whole aspect is slightly down-weighted, not just the
        exact element.

        Parameters
        ----------
        elements:
            Sequence of ``(aspect_name, element_uri)`` pairs.
        aspect_blend:
            How much the aspect-level penalty contributes.
            ``0.0`` = element-only, ``1.0`` = aspect-only.
        """
        if not elements:
            return []

        aspect_counts = self._aspect_counts()
        element_counts = self._element_counts()

        if not aspect_counts and not element_counts:
            return [1.0] * len(elements)

        weights: list[float] = []
        for aspect_name, element_uri in elements:
            elem_w = self._decay_weight(element_counts.get(element_uri, 0))
            asp_w = self._decay_weight(aspect_counts.get(aspect_name, 0))
            blended = (1.0 - aspect_blend) * elem_w + aspect_blend * asp_w
            weights.append(blended)

        return weights

    def _recent_uris(self) -> list[str]:
        """Extract all resource URIs from recent undo actions."""
        actions = self.undo_queue.peek_recent(self.peek_count)
        uris: list[str] = []
        for action in actions:
            uris.extend(self._extract_uris(action.method, action.params))
        return uris

    def _aspect_counts(self) -> Counter[str]:
        """Count how often each aspect appears in recent actions."""
        counter: Counter[str] = Counter()
        for uri in self._recent_uris():
            aspect = self._aspect_from_uri(uri)
            if aspect:
                counter[aspect] += 1
        return counter

    def _element_counts(self) -> Counter[str]:
        """Count how often each element URI appears in recent actions."""
        return Counter(self._recent_uris())

    def _decay_weight(self, count: int) -> float:
        """Convert an occurrence count into a positive weight in (0, 1]."""
        return 1.0 / (1.0 + count * self.decay)

    @staticmethod
    def _extract_uris(method: str, params: dict[str, Any]) -> list[str]:
        """Parse resource URIs from a single ReversibleAction."""
        uris: list[str] = []

        if method == "apply":
            op = params.get("op", {})
            target = op.get("target")
            if target == "property":
                uri = op.get("resource_uri")
                if uri:
                    uris.append(uri)
            elif target == "resource":
                loc = op.get("location", {})
                uri = loc.get("resource_uri")
                if uri:
                    uris.append(uri)

        elif method == "update_relationships":
            for key in ("source_uri", "target_uri"):
                uri = params.get(key)
                if uri:
                    uris.append(uri)

        elif method in ("remove_aspect", "add_aspect"):
            name = params.get("name")
            if name:
                uris.append(f"/{name}")

        return uris

    @staticmethod
    def _aspect_from_uri(uri: str) -> str | None:
        """Derive the aspect name from a resource URI like ``/character/foo``."""
        parts = uri.strip("/").split("/", 1)
        return parts[0] if parts and parts[0] else None
