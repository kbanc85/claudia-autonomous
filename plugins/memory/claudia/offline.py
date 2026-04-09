"""Three-tier offline degradation router (Phase 2A.2e).

Decides which ``HybridWeights`` preset the memory provider should
use based on runtime availability probes, and orchestrates the
common "embed query → run hybrid search" flow with graceful
mid-query degradation.

Three modes, matching the design doc:

+----------------+----------------------------+------------------------+
| Mode           | When                       | Weights                |
+================+============================+========================+
| FULL_HYBRID    | Ollama reachable, embedder | full_hybrid:           |
|                | can generate query vectors | 50 vec + 25 imp +      |
|                |                            | 10 rec + 15 fts +      |
|                |                            | 10 rehearsal           |
+----------------+----------------------------+------------------------+
| FTS_IMP_REC    | Ollama unreachable; the    | fts_imp_rec:           |
|                | vector component cannot    | 0 vec + 50 imp +       |
|                | contribute to scoring      | 20 rec + 30 fts +      |
|                |                            | 10 rehearsal           |
+----------------+----------------------------+------------------------+
| PURE_FTS       | Explicit override — user   | pure_fts:              |
|                | wants FTS-only ranking, or | 0 vec + 0 imp +        |
|                | no importance signal       | 0 rec + 100 fts +      |
|                | exists yet                 | 0 rehearsal            |
+----------------+----------------------------+------------------------+

How mode selection actually works:

``OllamaEmbedder`` caches its availability internally. Once
``embedder.is_available()`` returns True, subsequent calls return
True without re-probing. If an ``embed()`` call then fails mid-
session (connection dropped, daemon restarted), the embedder flips
its own ``_available`` flag to False. That flag IS the router's
signal — it does not need explicit failure notifications from the
caller. The router just calls ``select_mode()`` before each query
and trusts the embedder's state.

The one scenario that needs in-query handling is *transient failure
during a query that started in FULL_HYBRID mode*. The ``search()``
convenience method on the router handles this by re-selecting the
mode after a failed ``embed()`` call, so the composite scoring uses
the right weights even though the caller already committed to a
FULL_HYBRID path.

PURE_FTS is a deliberate escape hatch, not an automatic state.
There's nothing a runtime probe can reliably detect that says "use
pure FTS only" — vec0 extension presence is orthogonal (my
implementation uses pure-Python cosine), and zero-importance
memories are a usability problem, not a mode. PURE_FTS exists for
explicit opt-in via ``force_mode`` and for providers that want to
minimize signal during bootstrapping.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.2e)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from plugins.memory.claudia import hybrid_search
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.hybrid_search import (
    DEFAULT_RESULT_LIMIT,
    HybridWeights,
    SearchResult,
)

logger = logging.getLogger(__name__)


# ─── Mode enum ───────────────────────────────────────────────────────────


class MemoryMode(str, Enum):
    """The three runtime modes for Claudia's memory scoring.

    Subclassing ``str`` so that ``MemoryMode.FULL_HYBRID == "full_hybrid"``
    for convenient logging and serialization. The ordering of the
    members reflects the degradation path from most capable to least.
    """

    FULL_HYBRID = "full_hybrid"
    FTS_IMP_REC = "fts_imp_rec"
    PURE_FTS = "pure_fts"


#: Maps each mode to the ``HybridWeights`` preset factory.
_MODE_WEIGHTS = {
    MemoryMode.FULL_HYBRID: HybridWeights.full_hybrid,
    MemoryMode.FTS_IMP_REC: HybridWeights.fts_imp_rec,
    MemoryMode.PURE_FTS: HybridWeights.pure_fts,
}


# ─── Decision record ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModeDecision:
    """A single mode-selection outcome plus the context that produced it.

    Surfaced from ``OfflineRouter.select_mode()`` so callers can log
    transitions, stamp per-query metadata, or inspect why a downgrade
    happened without peering at the router's internals.
    """

    mode: MemoryMode
    weights: HybridWeights
    ollama_available: bool
    reason: str


# ─── Router ──────────────────────────────────────────────────────────────


class OfflineRouter:
    """Route memory queries through the right ``HybridWeights`` preset.

    Typical usage from ``provider.py`` (sub-task 2A.2f):

        embedder = OllamaEmbedder()
        router = OfflineRouter(embedder)
        # ... later, from prefetch() ...
        results = router.search(conn, query, profile=profile, limit=10)

    Or, if the caller wants to orchestrate embedding itself:

        decision = router.select_mode()
        if decision.mode == MemoryMode.FULL_HYBRID:
            blob, _ = embedder.embed(query) or (None, None)
        else:
            blob = None
        results = hybrid_search.search(
            conn, query, blob, weights=decision.weights, ...
        )

    The router is stateful only for mode-transition logging. All the
    real availability state lives inside the wrapped embedder.
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        *,
        force_mode: Optional[MemoryMode] = None,
    ) -> None:
        self._embedder = embedder
        self._force_mode = force_mode
        self._last_mode: Optional[MemoryMode] = None

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def embedder(self) -> Optional[OllamaEmbedder]:
        """The wrapped embedder, or None if the router is running embedderless."""
        return self._embedder

    @property
    def force_mode(self) -> Optional[MemoryMode]:
        """The forced mode override, if any."""
        return self._force_mode

    @property
    def last_mode(self) -> Optional[MemoryMode]:
        """The most recent mode returned by ``select_mode``, or None before first call."""
        return self._last_mode

    # ── Mode selection ────────────────────────────────────────────────

    def select_mode(self) -> ModeDecision:
        """Probe availability and return the mode + weights to use now.

        Cheap to call: when the embedder has cached its availability,
        this is a single attribute read. When force_mode is set, no
        probe happens at all.
        """
        if self._force_mode is not None:
            decision = self._build_decision(
                self._force_mode,
                ollama_available=(
                    self._embedder is not None and self._embedder.is_available()
                    if self._embedder is not None
                    else False
                ),
                reason=f"forced to {self._force_mode.value}",
            )
            self._log_transition(decision)
            return decision

        if self._embedder is None:
            decision = self._build_decision(
                MemoryMode.FTS_IMP_REC,
                ollama_available=False,
                reason="no embedder configured",
            )
            self._log_transition(decision)
            return decision

        if self._embedder.is_available():
            decision = self._build_decision(
                MemoryMode.FULL_HYBRID,
                ollama_available=True,
                reason="ollama available",
            )
        else:
            decision = self._build_decision(
                MemoryMode.FTS_IMP_REC,
                ollama_available=False,
                reason="ollama unavailable",
            )

        self._log_transition(decision)
        return decision

    def _build_decision(
        self,
        mode: MemoryMode,
        *,
        ollama_available: bool,
        reason: str,
    ) -> ModeDecision:
        weights = _MODE_WEIGHTS[mode]()
        return ModeDecision(
            mode=mode,
            weights=weights,
            ollama_available=ollama_available,
            reason=reason,
        )

    def _log_transition(self, decision: ModeDecision) -> None:
        if self._last_mode is not None and self._last_mode != decision.mode:
            logger.info(
                "Claudia memory mode transition: %s -> %s (%s)",
                self._last_mode.value,
                decision.mode.value,
                decision.reason,
            )
        self._last_mode = decision.mode

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear cached state and force a fresh probe on next select_mode.

        Useful at session boundaries or when an operator explicitly
        wants to re-check whether Ollama came back online. Also
        resets the ``last_mode`` sentinel so the next call doesn't
        log a spurious transition.
        """
        self._last_mode = None
        if self._embedder is not None:
            self._embedder.reset_availability()

    def set_force_mode(self, mode: Optional[MemoryMode]) -> None:
        """Override mode selection, or clear an existing override with None."""
        self._force_mode = mode
        # Don't clear last_mode — the next select_mode() will log the
        # transition naturally if the forced mode differs from the
        # runtime-probed mode.

    # ── One-shot search ───────────────────────────────────────────────

    def search(
        self,
        conn: sqlite3.Connection,
        query_text: str,
        *,
        profile: str = "default",
        limit: int = DEFAULT_RESULT_LIMIT,
        now: Optional[datetime] = None,
    ) -> List[SearchResult]:
        """Select mode, embed the query if needed, run hybrid search.

        This is the common path for providers: one call does
        everything. If the mode is FULL_HYBRID but the embed call
        then fails (transient Ollama failure mid-query), the router
        re-selects the mode and rebuilds the decision so the scoring
        weights match what was actually available.
        """
        decision = self.select_mode()
        query_embedding: Optional[bytes] = None

        if decision.mode == MemoryMode.FULL_HYBRID and self._embedder is not None:
            embed_result = self._embedder.embed(query_text)
            if embed_result is None:
                # Transient failure. The embedder has flipped its
                # internal availability flag; re-select so we use
                # FTS_IMP_REC weights for this query.
                logger.debug(
                    "Embed failed mid-query; re-selecting mode for %r",
                    query_text[:40],
                )
                decision = self.select_mode()
                query_embedding = None
            else:
                blob, _dim = embed_result
                query_embedding = blob

        return hybrid_search.search(
            conn,
            query_text,
            query_embedding=query_embedding,
            profile=profile,
            limit=limit,
            weights=decision.weights,
            now=now,
        )
