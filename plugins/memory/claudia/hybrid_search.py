"""Hybrid ranking for the Claudia memory provider (Phase 2A.2c).

Implements the composite scoring formula from the design doc:

    score = 50% vector + 25% importance + 10% recency + 15% FTS
          + rehearsal_boost * log(access_count)

Each component is normalized to [0, 1] before the weighted sum so
scores are comparable across queries. The rehearsal boost is
additive on top of the weighted base, favouring memories that have
been recalled often without displacing the four principal weights.

This module is deliberately independent of both Ollama and the
vec0 SQLite extension:

- Vector similarity is computed in Python against unpacked float32
  blobs from ``embeddings.unpack_embedding``. That means hybrid
  search works with only stdlib + sqlite3, no sqlite-vec import.
  When 2A.2e wires in vec0 for larger memory stores, it can replace
  the per-row Python cosine with a ``vec0`` MATCH query without
  touching callers.

- The caller is responsible for producing ``query_embedding`` via
  ``OllamaEmbedder.embed()``. If the caller passes ``None`` (Ollama
  offline, or the provider is in FTS-only mode), the vector
  component silently contributes 0 and the ``HybridWeights``
  preset handles the re-weighting (see ``HybridWeights.fts_imp_rec``
  and ``HybridWeights.pure_fts``).

Candidate gather strategy:

Hybrid search surfaces memories that would not be found by any
single signal alone. Candidates are accumulated from four sources
and then scored against the full composite:

1. **FTS5 MATCH** on the memories_fts virtual table. Picks up
   lexical matches, up to ``candidate_limit`` rows.
2. **Vector similarity scan** across stored embeddings when a
   query embedding is supplied. Catches semantic matches that have
   no lexical overlap.
3. **Top-by-importance safety net**. Ensures high-signal memories
   are always considered even when the query is narrow.
4. **Top-by-recency safety net**. Keeps the most recently accessed
   memories in the candidate pool so recent context survives.

The four candidate sets are UNION'd by memory id and the full row
is then scored once. For a profile with ~10k memories and
candidate_limit=200, this keeps each search at O(n) on the
embedding scan path but bounded on the scoring path.

Offline degradation (Phase 2A.2e preview):

``HybridWeights`` exposes three presets matching the design doc's
three-tier fallback:

- ``full_hybrid`` — 50/25/10/15 + rehearsal, used when Ollama is up
- ``fts_imp_rec`` — 0/50/20/30, used when Ollama is down
- ``pure_fts`` — 0/0/0/100, used when FTS is the only signal

2A.2e will select the right preset at runtime based on availability
probes and pass it to ``search()``. This module does not detect
availability itself; it just honours the weights it receives.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.2c)
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from plugins.memory.claudia.embeddings import unpack_embedding

logger = logging.getLogger(__name__)


# ─── Defaults ────────────────────────────────────────────────────────────

#: Default half-life for the recency exponential decay, in days. A
#: memory accessed 30 days ago scores 0.5 on the recency component,
#: 60 days ago scores 0.25, and so on. The design doc treats 30 days
#: as the baseline for "warm" memory; older memories still score but
#: have to earn their place via importance, vector similarity, or FTS.
DEFAULT_HALF_LIFE_DAYS = 30.0

#: Default number of top results returned by ``search()``.
DEFAULT_RESULT_LIMIT = 10

#: Default maximum number of candidates gathered per source (FTS,
#: vector, importance, recency). Each source contributes up to this
#: many ids into the UNION pool, so total scored rows is bounded by
#: roughly ``4 * candidate_limit``.
DEFAULT_CANDIDATE_LIMIT = 200


# ─── Trust weighting (Phase 2C.1) ────────────────────────────────────────

#: Verification-status multipliers. Applied to the final composite
#: score (base + boost) so the ordering reflects how much Claudia
#: trusts the memory, not just how relevant it is lexically or
#: semantically. The trust north star principle says low-confidence
#: and contradicted memories should rank lower; this is the
#: ranking-side enforcement.
#:
#: - ``verified``: explicitly confirmed. Full weight.
#: - ``pending``: default state for newly-stored memories. Not yet
#:   verified, but nothing suggests a problem. No penalty — a fresh
#:   extracted memory ranks with its full composite score.
#: - ``flagged``: marked as "needs review" (by the user or by the
#:   verification service's stale-flagging pass). Halved — still
#:   reachable via recall but demoted so verified memories win ties.
#: - ``contradicts``: explicitly flagged as conflicting with another
#:   memory. Strong demotion to keep the contradicted memory out of
#:   prefetch output unless nothing better matches.
VERIFICATION_WEIGHTS = {
    "verified": 1.0,
    "pending": 1.0,
    "flagged": 0.5,
    "contradicts": 0.3,
}

#: Fallback multiplier for unrecognized verification values. Schema
#: has a CHECK constraint so this should never fire in practice,
#: but defensive: unknown status gets the "treat as pending" default
#: rather than silently excluded or boosted.
DEFAULT_VERIFICATION_WEIGHT = 1.0


def _verification_weight(status: Optional[str]) -> float:
    """Return the multiplier for a verification status string.

    Falls back to ``DEFAULT_VERIFICATION_WEIGHT`` for None, empty,
    or unknown values so a stray NULL never crashes the scoring loop.
    """
    if not status:
        return DEFAULT_VERIFICATION_WEIGHT
    return VERIFICATION_WEIGHTS.get(status, DEFAULT_VERIFICATION_WEIGHT)


# ─── Weight presets ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class HybridWeights:
    """Composite scoring weights for hybrid search.

    The four principal components (``vec``, ``importance``, ``recency``,
    ``fts``) form the weighted base and are expected to sum to 1.0 in
    normal operation. ``rehearsal_boost`` is additive on top of the
    base — it boosts frequently-accessed memories without displacing
    the base weights.

    Use the class method presets for the three offline modes documented
    in the design doc; construct directly only for tests or custom
    tuning during Phase 2B hyperparameter sweeps.
    """

    vec: float = 0.50
    importance: float = 0.25
    recency: float = 0.10
    fts: float = 0.15
    rehearsal_boost: float = 0.10

    @classmethod
    def full_hybrid(cls) -> "HybridWeights":
        """Default 50/25/10/15 + 10% rehearsal. Ollama up, vec0 optional."""
        return cls()

    @classmethod
    def fts_imp_rec(cls) -> "HybridWeights":
        """Ollama offline path. Drop vector, reweight to 50/20/30 + rehearsal.

        When the embedder cannot generate new embeddings, vector
        similarity becomes meaningless (we can't compare the query
        against memory embeddings without a query embedding). The vec
        weight goes to 0 and the remaining three components absorb its
        share, with importance taking the largest chunk to preserve
        the signal Claudia trusts most from the user.
        """
        return cls(
            vec=0.0,
            importance=0.50,
            recency=0.20,
            fts=0.30,
            rehearsal_boost=0.10,
        )

    @classmethod
    def pure_fts(cls) -> "HybridWeights":
        """Bottom tier. FTS only, no rehearsal boost.

        Used when neither vec0 nor the embedder are available and
        the memory store was created without importance metadata.
        A pure-FTS search is equivalent to a naive ``bm25()`` ranking
        but still respects profile isolation and soft deletes.
        """
        return cls(
            vec=0.0,
            importance=0.0,
            recency=0.0,
            fts=1.0,
            rehearsal_boost=0.0,
        )


# ─── Result types ────────────────────────────────────────────────────────


@dataclass
class ScoreBreakdown:
    """Per-memory component scores for debugging and explainability.

    Each signal field is in [0, 1] after normalization. ``total`` is
    the composite after ``HybridWeights``, the rehearsal boost, AND
    the Phase 2C.1 trust multiplier (confidence × verification weight).
    It can exceed 1.0 when a memory has a strong rehearsal history on
    top of a full weighted base and a trust factor of 1.0.

    Trust fields:
      - ``confidence``: memories.confidence column, [0, 1]. Reflects
        certainty at storage + 2B.4 decay.
      - ``verification``: memories.verification column value.
      - ``verification_multiplier``: numeric weight from
        ``VERIFICATION_WEIGHTS`` for the status.
      - ``trust_factor``: confidence × verification_multiplier. This
        is what ``total`` was multiplied by before being returned.
    """

    vec: float = 0.0
    importance: float = 0.0
    recency: float = 0.0
    fts: float = 0.0
    rehearsal: float = 0.0
    total: float = 0.0
    confidence: float = 1.0
    verification: str = "pending"
    verification_multiplier: float = 1.0
    trust_factor: float = 1.0


@dataclass
class SearchResult:
    """A single scored memory returned from hybrid search.

    ``score`` is the post-trust-factor composite: the raw weighted
    base plus rehearsal boost, multiplied by
    ``confidence × verification_multiplier``. A high-relevance
    memory with low trust can rank below a moderately-relevant
    memory with high trust.
    """

    memory_id: int
    content: str
    score: float
    breakdown: ScoreBreakdown
    importance: float
    access_count: int
    entity_id: Optional[int]
    created_at: str
    accessed_at: str
    source_type: Optional[str]
    source_ref: Optional[str]
    confidence: float
    verification: str = "pending"


# ─── Vector math helper ─────────────────────────────────────────────────


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity between two float vectors.

    Returns 0.0 on empty input, mismatched dimensions, or zero
    vectors. The valid range is [-1, 1] for non-degenerate input,
    but callers should usually clamp the result to [0, 1] before
    combining with other normalized components — negative cosines
    typically indicate "opposite" meaning and shouldn't be treated
    as signal.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ─── FTS query construction ─────────────────────────────────────────────


# Regex to split a query into FTS5-safe tokens. Matches runs of word
# characters (Unicode-aware: letters, digits, underscore, plus any
# letter-like codepoint). Everything else is treated as a separator.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def build_fts_query(text: str) -> str:
    """Transform user text into an FTS5 MATCH expression.

    FTS5's default boolean operator is AND, which is too restrictive
    for recall-oriented hybrid search. We tokenize the query, drop
    single-character tokens (which are mostly noise), and join with
    explicit ``OR`` so that any matching token contributes a candidate.

    Returns an empty string when the input has no usable tokens;
    callers should skip the FTS gather in that case.
    """
    if not text:
        return ""

    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]
    if not tokens:
        return ""

    # Dedupe while preserving order so the query is deterministic.
    seen: set = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return " OR ".join(unique)


# ─── Candidate gather ───────────────────────────────────────────────────


def _gather_fts_candidates(
    conn: sqlite3.Connection,
    fts_query: str,
    profile: str,
    limit: int,
) -> Dict[int, float]:
    """Run FTS5 MATCH and return {memory_id: raw_bm25_rank}.

    ``bm25()`` returns a negative number where smaller (more negative)
    = better match. The raw values are returned here so the caller can
    min-max normalize across the candidate set in the scoring pass.
    """
    if not fts_query:
        return {}

    try:
        rows = conn.execute(
            """
            SELECT mf.rowid AS id, bm25(memories_fts) AS rank
            FROM memories_fts mf
            JOIN memories m ON m.id = mf.rowid
            WHERE memories_fts MATCH ?
              AND m.profile = ?
              AND m.deleted_at IS NULL
            ORDER BY bm25(memories_fts)
            LIMIT ?
            """,
            (fts_query, profile, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # Malformed FTS query (shouldn't happen after build_fts_query,
        # but defend against it). Log and return empty rather than
        # crashing the caller's prefetch path.
        logger.debug("FTS MATCH failed for %r: %s", fts_query, exc)
        return {}

    return {row["id"]: float(row["rank"]) for row in rows}


def _gather_top_importance(
    conn: sqlite3.Connection, profile: str, limit: int
) -> List[int]:
    rows = conn.execute(
        """
        SELECT id FROM memories
        WHERE profile = ? AND deleted_at IS NULL
        ORDER BY importance DESC, accessed_at DESC
        LIMIT ?
        """,
        (profile, limit),
    ).fetchall()
    return [row["id"] for row in rows]


def _gather_top_recency(
    conn: sqlite3.Connection, profile: str, limit: int
) -> List[int]:
    rows = conn.execute(
        """
        SELECT id FROM memories
        WHERE profile = ? AND deleted_at IS NULL
        ORDER BY accessed_at DESC
        LIMIT ?
        """,
        (profile, limit),
    ).fetchall()
    return [row["id"] for row in rows]


def _gather_vector_candidates(
    conn: sqlite3.Connection,
    query_vec: List[float],
    profile: str,
    limit: int,
) -> Dict[int, float]:
    """Scan stored embeddings and return top-``limit`` by cosine similarity.

    Returns {memory_id: cosine_similarity_in_[0, 1]}. Skips memories
    whose stored embedding dimension does not match the query dim —
    those would indicate a model swap that invalidated old embeddings
    (see Claudia v1's ``--migrate-embeddings`` workflow for the full
    migration path; the plugin's own migration story is Phase 2A.2f).
    """
    if not query_vec:
        return {}

    rows = conn.execute(
        """
        SELECT id, embedding, embedding_dim FROM memories
        WHERE profile = ?
          AND deleted_at IS NULL
          AND embedding IS NOT NULL
        """,
        (profile,),
    ).fetchall()

    query_dim = len(query_vec)
    sims: List[tuple] = []
    for row in rows:
        stored_dim = row["embedding_dim"]
        if stored_dim is not None and stored_dim != query_dim:
            continue
        mem_vec = unpack_embedding(row["embedding"])
        if len(mem_vec) != query_dim:
            # Defensive: stored dim column disagrees with actual blob.
            continue
        sim = cosine_similarity(query_vec, mem_vec)
        # Clamp to [0, 1]; negative cosines contribute nothing.
        sim = max(0.0, sim)
        if sim > 0:
            sims.append((row["id"], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return {mid: sim for mid, sim in sims[:limit]}


# ─── Scoring ─────────────────────────────────────────────────────────────


def _recency_score(accessed_at_iso: str, now: datetime, half_life_days: float) -> float:
    """Exponential decay: 1.0 at age 0, 0.5 at age=half_life, etc."""
    try:
        accessed_at = datetime.fromisoformat(accessed_at_iso)
    except (ValueError, TypeError):
        return 0.0

    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=timezone.utc)

    age_seconds = (now - accessed_at).total_seconds()
    if age_seconds <= 0:
        return 1.0

    age_days = age_seconds / 86400.0
    decay = math.exp(-age_days * math.log(2) / half_life_days)
    # Guard against pathological floats
    return max(0.0, min(1.0, decay))


def _normalize_fts_ranks(raw_ranks: Dict[int, float]) -> Dict[int, float]:
    """Map bm25 raw scores (negative, lower=better) to [0, 1].

    If all candidates have the same raw rank (or only one candidate),
    every FTS component collapses to 1.0 since the ordering carries
    no information.
    """
    if not raw_ranks:
        return {}

    values = list(raw_ranks.values())
    min_raw = min(values)  # most negative = best match
    max_raw = max(values)  # least negative
    span = max_raw - min_raw

    if span == 0:
        return {mid: 1.0 for mid in raw_ranks}

    return {
        mid: (max_raw - raw) / span
        for mid, raw in raw_ranks.items()
    }


# ─── Public API ──────────────────────────────────────────────────────────


def search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: Optional[bytes] = None,
    *,
    profile: str = "default",
    limit: int = DEFAULT_RESULT_LIMIT,
    weights: Optional[HybridWeights] = None,
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> List[SearchResult]:
    """Run hybrid search and return up to ``limit`` ranked memories.

    Parameters
    ----------
    conn:
        Open sqlite3 connection to a Claudia memory database (as
        returned by ``schema.ensure_database``).
    query_text:
        The user query. Used for FTS gather and for informative
        empty-query handling. An empty string is valid and falls
        back to importance + recency gathers.
    query_embedding:
        Little-endian float32 bytes from
        ``embeddings.OllamaEmbedder.embed()``, or ``None`` when the
        embedder is offline / the caller is running in an FTS-only
        mode. When ``None``, the vector component contributes 0 and
        the vector-gather step is skipped.
    profile:
        Profile isolation key. Matches the ``profile`` column on
        the memories table. Default is ``"default"``.
    limit:
        Maximum number of results returned to the caller. The
        internal candidate pool is much larger (see
        ``candidate_limit``).
    weights:
        ``HybridWeights`` preset or custom instance. Defaults to
        ``HybridWeights.full_hybrid()``. Pass a different preset to
        run one of the offline modes.
    now:
        Reference time for recency scoring. Defaults to
        ``datetime.now(timezone.utc)``. Tests should pin this for
        deterministic recency assertions.
    half_life_days:
        Half-life for the recency exponential decay. Default 30.
    candidate_limit:
        Maximum candidates gathered from each source. Total scored
        rows is bounded by roughly ``4 * candidate_limit``.

    Returns
    -------
    List[SearchResult]
        Sorted by composite score descending. Empty list if the
        profile has no matching memories.
    """
    now = now or datetime.now(timezone.utc)
    weights = weights or HybridWeights.full_hybrid()
    query_text = (query_text or "").strip()

    # ── Gather candidates ─────────────────────────────────────────────

    fts_query = build_fts_query(query_text)
    fts_raw_ranks = _gather_fts_candidates(conn, fts_query, profile, candidate_limit)

    query_vec: List[float] = (
        unpack_embedding(query_embedding) if query_embedding else []
    )
    vec_sims = (
        _gather_vector_candidates(conn, query_vec, profile, candidate_limit)
        if query_vec
        else {}
    )

    importance_ids = _gather_top_importance(conn, profile, candidate_limit)
    recency_ids = _gather_top_recency(conn, profile, candidate_limit)

    candidate_ids: set = set()
    candidate_ids.update(fts_raw_ranks.keys())
    candidate_ids.update(vec_sims.keys())
    candidate_ids.update(importance_ids)
    candidate_ids.update(recency_ids)

    if not candidate_ids:
        return []

    # ── Fetch full rows for scoring ───────────────────────────────────

    placeholders = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        f"""
        SELECT id, content, entity_id, origin, confidence, importance,
               access_count, embedding, embedding_dim, source_type,
               source_ref, created_at, accessed_at, verification
        FROM memories
        WHERE id IN ({placeholders})
          AND profile = ?
          AND deleted_at IS NULL
        """,
        (*candidate_ids, profile),
    ).fetchall()

    if not rows:
        return []

    # ── Normalize cross-row signals ───────────────────────────────────

    fts_scores = _normalize_fts_ranks(fts_raw_ranks)

    # Rehearsal boost: log-normalized access_count across the candidate
    # set. Using the candidate max (not the global max) keeps the boost
    # meaningful for every query — a moderately-rehearsed memory still
    # beats a cold one even when the candidate pool is all warm.
    max_access = max((row["access_count"] for row in rows), default=0)
    log_max_access = math.log(max_access + 1) if max_access > 0 else 0.0

    # ── Score each candidate ─────────────────────────────────────────

    results: List[SearchResult] = []
    query_dim = len(query_vec)

    for row in rows:
        # Vector component — prefer the pre-computed gather sim
        # (already clamped to [0, 1]) when available, otherwise
        # compute on the fly for memories that came in via other
        # gather paths but still have an embedding.
        if row["id"] in vec_sims:
            vec_score = vec_sims[row["id"]]
        elif query_vec and row["embedding"] and row["embedding_dim"] == query_dim:
            mem_vec = unpack_embedding(row["embedding"])
            if len(mem_vec) == query_dim:
                vec_score = max(0.0, cosine_similarity(query_vec, mem_vec))
            else:
                vec_score = 0.0
        else:
            vec_score = 0.0

        imp_score = float(row["importance"])
        rec_score = _recency_score(row["accessed_at"], now, half_life_days)
        fts_score = fts_scores.get(row["id"], 0.0)

        if log_max_access > 0:
            rehearsal_score = math.log(row["access_count"] + 1) / log_max_access
        else:
            rehearsal_score = 0.0

        base = (
            weights.vec * vec_score
            + weights.importance * imp_score
            + weights.recency * rec_score
            + weights.fts * fts_score
        )
        boost = weights.rehearsal_boost * rehearsal_score
        raw_total = base + boost

        # Phase 2C.1: apply the trust factor. Confidence is clamped
        # to [0, 1] defensively in case of a schema violation;
        # verification weight comes from the constant table with a
        # default for unknown statuses.
        confidence = float(row["confidence"])
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0

        verification_status = row["verification"] or "pending"
        verification_multiplier = _verification_weight(verification_status)
        trust_factor = confidence * verification_multiplier
        total = raw_total * trust_factor

        breakdown = ScoreBreakdown(
            vec=vec_score,
            importance=imp_score,
            recency=rec_score,
            fts=fts_score,
            rehearsal=rehearsal_score,
            total=total,
            confidence=confidence,
            verification=verification_status,
            verification_multiplier=verification_multiplier,
            trust_factor=trust_factor,
        )

        results.append(
            SearchResult(
                memory_id=row["id"],
                content=row["content"],
                score=total,
                breakdown=breakdown,
                importance=imp_score,
                access_count=row["access_count"],
                entity_id=row["entity_id"],
                created_at=row["created_at"],
                accessed_at=row["accessed_at"],
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                confidence=confidence,
                verification=verification_status,
            )
        )

    # ── Sort and return top ``limit`` ────────────────────────────────

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
