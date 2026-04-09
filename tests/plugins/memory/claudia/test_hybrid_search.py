"""Unit tests for plugins/memory/claudia/hybrid_search.py (Phase 2A.2c).

Covers:

- ``HybridWeights`` presets (full_hybrid, fts_imp_rec, pure_fts)
- ``cosine_similarity`` edge cases: empty, mismatched dims, zero vectors
- ``build_fts_query`` tokenization and OR-joining
- FTS-only match ranking via ``search()`` with ``pure_fts`` weights
- Vector-only match ranking using a real query embedding
- Importance ranking dominance when weights favour it
- Recency exponential decay via pinned ``now``
- Rehearsal boost from access_count
- Composite ordering across multiple signals
- Profile isolation (memories in other profiles invisible)
- Soft-delete exclusion (deleted_at IS NULL filter)
- Empty query with full_hybrid weights
- Query embedding with dimension mismatch is silently ignored
- Empty candidate pool returns empty list

Tests use a real SQLite database via ``schema.ensure_database`` so we
exercise the FTS5 MATCH + triggers end-to-end. The ``_isolate_claudia_home``
fixture in tests/conftest.py redirects CLAUDIA_HOME to ``tmp_path``,
so these tests never write to the real ~/.claudia.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from plugins.memory.claudia.embeddings import pack_embedding, unpack_embedding
from plugins.memory.claudia.hybrid_search import (
    DEFAULT_HALF_LIFE_DAYS,
    HybridWeights,
    ScoreBreakdown,
    SearchResult,
    build_fts_query,
    cosine_similarity,
    search,
)
from plugins.memory.claudia.schema import ensure_database


# ─── Fixtures and helpers ────────────────────────────────────────────────


@pytest.fixture()
def memory_db(tmp_path):
    """Fresh Claudia memory DB at a tmp path, schema applied, WAL on."""
    db_path = tmp_path / "claudia.db"
    conn = ensure_database(db_path)
    yield conn
    conn.close()


# A fixed "now" anchor so recency tests are deterministic.
NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def insert_memory(
    conn,
    content: str,
    *,
    importance: float = 0.5,
    access_count: int = 0,
    embedding: Optional[bytes] = None,
    embedding_dim: Optional[int] = None,
    profile: str = "default",
    source_type: str = "test",
    source_ref: str = "test-1",
    origin: str = "user_stated",
    confidence: float = 0.9,
    verification: str = "pending",
    created_at: Optional[datetime] = None,
    accessed_at: Optional[datetime] = None,
    deleted_at: Optional[datetime] = None,
) -> int:
    """Insert a memory row and return its rowid.

    Defaults keep tests terse — override only the fields a given test
    cares about. ``created_at`` and ``accessed_at`` default to NOW so
    recency scoring isn't accidentally influenced by fresh-timestamp
    behaviour unless the test pins it explicitly.
    """
    created_at = created_at or NOW
    accessed_at = accessed_at or NOW

    cur = conn.execute(
        """
        INSERT INTO memories (
            content, importance, access_count, embedding, embedding_dim,
            profile, source_type, source_ref, origin, confidence,
            verification, created_at, accessed_at, deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            importance,
            access_count,
            embedding,
            embedding_dim,
            profile,
            source_type,
            source_ref,
            origin,
            confidence,
            verification,
            created_at.isoformat(),
            accessed_at.isoformat(),
            deleted_at.isoformat() if deleted_at else None,
        ),
    )
    return cur.lastrowid


# A tiny, hand-picked set of 3-dim vectors. Their cosine similarities
# are easy to reason about:
#   [1, 0, 0] · [1, 0, 0] = 1    (identical)
#   [1, 0, 0] · [0, 1, 0] = 0    (orthogonal)
#   [1, 0, 0] · [0.9, 0.1, 0] ~ 0.994 (close)
VEC_EAST = pack_embedding([1.0, 0.0, 0.0])
VEC_NORTH = pack_embedding([0.0, 1.0, 0.0])
VEC_NE = pack_embedding([0.9, 0.1, 0.0])
VEC_WEST = pack_embedding([-1.0, 0.0, 0.0])


# ─── HybridWeights presets ───────────────────────────────────────────────


class TestHybridWeightsPresets:
    def test_full_hybrid_defaults(self):
        w = HybridWeights.full_hybrid()
        assert w.vec == 0.50
        assert w.importance == 0.25
        assert w.recency == 0.10
        assert w.fts == 0.15
        assert w.rehearsal_boost == 0.10
        # Principal components sum to 1.0
        assert w.vec + w.importance + w.recency + w.fts == pytest.approx(1.0)

    def test_fts_imp_rec_drops_vector(self):
        w = HybridWeights.fts_imp_rec()
        assert w.vec == 0.0
        # The three remaining components take over the full share
        assert w.importance + w.recency + w.fts == pytest.approx(1.0)
        # Importance still dominates — Claudia trusts user signal most
        assert w.importance > w.fts > w.recency

    def test_pure_fts_is_fts_only(self):
        w = HybridWeights.pure_fts()
        assert w.vec == 0.0
        assert w.importance == 0.0
        assert w.recency == 0.0
        assert w.fts == 1.0
        assert w.rehearsal_boost == 0.0


# ─── cosine_similarity ──────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_returns_zero(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_mismatched_dims_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_roundtrip_via_pack_unpack(self):
        """Sanity check: pack + unpack preserves cosine similarity."""
        a = [0.5, 0.3, 0.2]
        b = [0.1, 0.9, 0.0]
        expected = cosine_similarity(a, b)
        actual = cosine_similarity(
            unpack_embedding(pack_embedding(a)),
            unpack_embedding(pack_embedding(b)),
        )
        assert actual == pytest.approx(expected, rel=1e-5)


# ─── build_fts_query ────────────────────────────────────────────────────


class TestBuildFtsQuery:
    def test_empty_input_returns_empty(self):
        assert build_fts_query("") == ""
        assert build_fts_query(None) == ""  # type: ignore[arg-type]

    def test_whitespace_only_returns_empty(self):
        assert build_fts_query("   \n\t  ") == ""

    def test_single_char_tokens_dropped(self):
        # "said" and "hello" survive; "i" and "a" are too short to be signal
        q = build_fts_query("i said a hello")
        assert "said" in q
        assert "hello" in q
        # Single-char tokens dropped — watch out for substring false positives
        tokens = q.replace(" OR ", " ").split()
        assert "i" not in tokens
        assert "a" not in tokens

    def test_tokens_ored_together(self):
        q = build_fts_query("sarah proposal")
        assert "OR" in q
        assert "sarah" in q
        assert "proposal" in q

    def test_punctuation_stripped(self):
        q = build_fts_query("what did Sarah say about the proposal?")
        assert "?" not in q
        assert "sarah" in q
        assert "proposal" in q

    def test_lowercase_normalization(self):
        q = build_fts_query("SARAH")
        assert "SARAH" not in q
        assert "sarah" in q

    def test_dedupes_repeated_tokens(self):
        q = build_fts_query("sarah sarah sarah")
        # Should contain "sarah" exactly once
        assert q.count("sarah") == 1


# ─── Pure-FTS search ────────────────────────────────────────────────────


class TestPureFtsSearch:
    def test_lexical_match_returns_row(self, memory_db):
        insert_memory(memory_db, "Sarah promised the proposal by Friday")
        insert_memory(memory_db, "Unrelated note about pelicans")

        results = search(
            memory_db,
            "sarah proposal",
            weights=HybridWeights.pure_fts(),
            now=NOW,
        )

        assert len(results) >= 1
        # Sarah memory should be ranked first
        assert "Sarah" in results[0].content
        # Raw component scores are recorded regardless of weights:
        # breakdown.fts is normalized bm25, breakdown.importance is the
        # row's raw importance. The weights only affect the composite total.
        assert results[0].breakdown.fts > 0
        assert results[0].breakdown.vec == 0.0  # no embeddings stored
        # With pure_fts weights, the base composite equals the fts
        # component only (other weights are zero). The final score is
        # then multiplied by the trust factor (Phase 2C.1):
        # score == fts * confidence * verification_multiplier.
        expected = (
            results[0].breakdown.fts * results[0].breakdown.trust_factor
        )
        assert results[0].score == pytest.approx(expected)

    def test_no_lexical_match_falls_back_to_importance(self, memory_db):
        """When FTS finds nothing, the importance safety net still returns results."""
        insert_memory(memory_db, "High-importance note", importance=0.9)
        insert_memory(memory_db, "Lower importance note", importance=0.3)

        results = search(
            memory_db,
            "xyzzy",  # no matches
            weights=HybridWeights.full_hybrid(),
            now=NOW,
        )

        # The safety net pulls importance candidates — non-empty result
        assert len(results) == 2
        # In full_hybrid mode, importance>fts so high-imp wins
        assert results[0].importance == 0.9


# ─── Vector-based search ────────────────────────────────────────────────


class TestVectorSearch:
    def test_semantic_match_beats_lexical_mismatch(self, memory_db):
        """Query embedding close to memory A's embedding; lexical hits memory B."""
        # Memory A: lexical content is irrelevant; its embedding points east
        insert_memory(
            memory_db,
            "zonk quibbles flumph",  # no lexical overlap with query
            embedding=VEC_EAST,
            embedding_dim=3,
            importance=0.5,
        )
        # Memory B: lexical match but vector points north (orthogonal)
        insert_memory(
            memory_db,
            "east east east",  # lexical match to query "east"
            embedding=VEC_NORTH,
            embedding_dim=3,
            importance=0.5,
        )

        # Query vector points east; with 50% vector weight, memory A wins
        results = search(
            memory_db,
            "east",
            query_embedding=VEC_EAST,
            weights=HybridWeights.full_hybrid(),
            now=NOW,
        )

        assert len(results) == 2
        # Memory A has vec_score ~= 1.0, memory B has vec_score ~= 0.0
        # Memory B has fts_score ~= 1.0 but only 0.15 weight; vec weighs 0.50
        # → A should win
        content_order = [r.content for r in results]
        assert content_order[0] == "zonk quibbles flumph"

    def test_cosine_similarity_in_breakdown(self, memory_db):
        insert_memory(
            memory_db,
            "doc 1",
            embedding=VEC_EAST,
            embedding_dim=3,
        )
        insert_memory(
            memory_db,
            "doc 2",
            embedding=VEC_NE,  # very close to east
            embedding_dim=3,
        )
        insert_memory(
            memory_db,
            "doc 3",
            embedding=VEC_WEST,  # opposite
            embedding_dim=3,
        )

        results = search(
            memory_db,
            "anything",
            query_embedding=VEC_EAST,
            weights=HybridWeights.full_hybrid(),
            now=NOW,
        )

        by_content = {r.content: r for r in results}
        assert by_content["doc 1"].breakdown.vec == pytest.approx(1.0, rel=1e-5)
        assert by_content["doc 2"].breakdown.vec == pytest.approx(
            0.994, abs=1e-2
        )
        # Opposite → cosine is -1, clamped to 0
        assert by_content["doc 3"].breakdown.vec == 0.0

    def test_dimension_mismatch_ignored(self, memory_db):
        """Stored embedding with wrong dim contributes 0 to vec score."""
        # 3-dim query embedding
        query_emb = pack_embedding([1.0, 0.0, 0.0])
        # Memory has a 4-dim embedding (simulates model migration)
        mem_emb = pack_embedding([1.0, 0.0, 0.0, 0.0])

        insert_memory(
            memory_db,
            "mismatched dim",
            embedding=mem_emb,
            embedding_dim=4,
            importance=0.5,
        )

        results = search(
            memory_db,
            "mismatched",
            query_embedding=query_emb,
            now=NOW,
        )

        # Still returns the row (via FTS gather), but vec component is 0
        assert len(results) == 1
        assert results[0].breakdown.vec == 0.0

    def test_no_query_embedding_means_zero_vec_component(self, memory_db):
        insert_memory(
            memory_db,
            "some content",
            embedding=VEC_EAST,
            embedding_dim=3,
            importance=0.5,
        )

        results = search(
            memory_db,
            "some",
            query_embedding=None,  # offline case
            now=NOW,
        )

        assert len(results) == 1
        assert results[0].breakdown.vec == 0.0


# ─── Importance ranking ─────────────────────────────────────────────────


class TestImportanceRanking:
    def test_high_importance_beats_low_when_no_other_signal(self, memory_db):
        insert_memory(memory_db, "boring thing", importance=0.1)
        insert_memory(memory_db, "critical thing", importance=0.95)
        insert_memory(memory_db, "medium thing", importance=0.5)

        results = search(
            memory_db,
            "",  # no query → FTS contributes nothing
            query_embedding=None,
            weights=HybridWeights.full_hybrid(),
            now=NOW,
        )

        assert len(results) == 3
        # Order: critical, medium, boring
        assert results[0].content == "critical thing"
        assert results[1].content == "medium thing"
        assert results[2].content == "boring thing"


# ─── Recency decay ──────────────────────────────────────────────────────


class TestRecencyDecay:
    def test_recent_memory_scores_higher_than_old(self, memory_db):
        insert_memory(
            memory_db,
            "ancient",
            importance=0.5,
            accessed_at=NOW - timedelta(days=90),  # ~8% recency
        )
        insert_memory(
            memory_db,
            "fresh",
            importance=0.5,
            accessed_at=NOW,  # 100% recency
        )

        results = search(
            memory_db,
            "",
            weights=HybridWeights.full_hybrid(),
            now=NOW,
        )

        assert len(results) == 2
        # Fresh wins on recency when importance is tied
        assert results[0].content == "fresh"
        assert results[0].breakdown.recency == pytest.approx(1.0)
        # 90 days = 3 half-lives → 0.125
        assert results[1].breakdown.recency == pytest.approx(0.125, abs=0.01)

    def test_half_life_exact(self, memory_db):
        insert_memory(
            memory_db,
            "half-life test",
            accessed_at=NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS),
        )

        results = search(memory_db, "", now=NOW)
        assert results[0].breakdown.recency == pytest.approx(0.5, abs=1e-4)


# ─── Rehearsal boost ─────────────────────────────────────────────────────


class TestRehearsalBoost:
    def test_rehearsed_memory_gets_boost(self, memory_db):
        insert_memory(memory_db, "cold", importance=0.5, access_count=0)
        insert_memory(memory_db, "warm", importance=0.5, access_count=50)

        results = search(memory_db, "", now=NOW)

        by_content = {r.content: r for r in results}
        assert by_content["warm"].breakdown.rehearsal == pytest.approx(1.0)
        assert by_content["cold"].breakdown.rehearsal == 0.0
        # Warm should outrank cold even though importance is tied
        assert results[0].content == "warm"

    def test_no_rehearsal_when_all_cold(self, memory_db):
        insert_memory(memory_db, "a", importance=0.5, access_count=0)
        insert_memory(memory_db, "b", importance=0.5, access_count=0)

        results = search(memory_db, "", now=NOW)

        for r in results:
            assert r.breakdown.rehearsal == 0.0


# ─── Profile isolation ──────────────────────────────────────────────────


class TestProfileIsolation:
    def test_other_profile_invisible(self, memory_db):
        insert_memory(memory_db, "alice memory", profile="alice", importance=0.9)
        insert_memory(memory_db, "bob memory", profile="bob", importance=0.9)

        alice_results = search(memory_db, "", profile="alice", now=NOW)
        bob_results = search(memory_db, "", profile="bob", now=NOW)

        assert len(alice_results) == 1
        assert alice_results[0].content == "alice memory"

        assert len(bob_results) == 1
        assert bob_results[0].content == "bob memory"

    def test_default_profile(self, memory_db):
        insert_memory(memory_db, "default memory", profile="default")
        insert_memory(memory_db, "scoped memory", profile="workspace-a")

        results = search(memory_db, "", now=NOW)  # profile defaults to "default"

        assert len(results) == 1
        assert results[0].content == "default memory"


# ─── Soft-delete exclusion ──────────────────────────────────────────────


class TestSoftDelete:
    def test_deleted_memory_not_returned(self, memory_db):
        insert_memory(memory_db, "alive", importance=0.5)
        insert_memory(
            memory_db,
            "dead",
            importance=0.9,  # would win if not deleted
            deleted_at=NOW,
        )

        results = search(memory_db, "", now=NOW)

        assert len(results) == 1
        assert results[0].content == "alive"

    def test_deleted_memory_not_found_by_fts(self, memory_db):
        insert_memory(memory_db, "unique_marker_token", deleted_at=NOW)
        insert_memory(memory_db, "other content")

        results = search(
            memory_db,
            "unique_marker_token",
            weights=HybridWeights.pure_fts(),
            now=NOW,
        )

        # Deleted row is filtered out in the FTS gather
        contents = [r.content for r in results]
        assert "unique_marker_token" not in contents


# ─── Empty and edge cases ───────────────────────────────────────────────


class TestEmptyAndEdgeCases:
    def test_empty_db_returns_empty_list(self, memory_db):
        results = search(memory_db, "anything", now=NOW)
        assert results == []

    def test_empty_query_returns_importance_ordered(self, memory_db):
        insert_memory(memory_db, "a", importance=0.3)
        insert_memory(memory_db, "b", importance=0.8)
        insert_memory(memory_db, "c", importance=0.1)

        results = search(memory_db, "", limit=10, now=NOW)

        assert len(results) == 3
        # Sorted by composite; with no FTS or vec, importance dominates
        assert [r.content for r in results] == ["b", "a", "c"]

    def test_limit_respected(self, memory_db):
        for i in range(10):
            insert_memory(memory_db, f"memory {i}", importance=i / 10.0)

        results = search(memory_db, "", limit=3, now=NOW)

        assert len(results) == 3

    def test_malformed_accessed_at_does_not_crash(self, memory_db):
        # Directly craft a row with bad timestamp — defensive check
        memory_db.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                profile, created_at, accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("bad ts", "user_stated", 0.9, 0.5, 0, "default", NOW.isoformat(), "not-a-timestamp"),
        )

        results = search(memory_db, "", now=NOW)
        # Should return the row with recency=0, not crash
        assert len(results) == 1
        assert results[0].breakdown.recency == 0.0


# ─── Phase 2C.1: verification-status weighting ──────────────────────────


class TestVerificationWeighting:
    """Flagged and contradicted memories should rank lower than
    verified and pending memories with the same base score."""

    def test_verified_ranks_equal_to_pending_on_same_base(self, memory_db):
        """Default pending and explicit verified both have multiplier 1.0."""
        insert_memory(
            memory_db, "pending fact about widgets",
            importance=0.8, confidence=1.0, verification="pending",
        )
        insert_memory(
            memory_db, "verified fact about widgets",
            importance=0.8, confidence=1.0, verification="verified",
        )
        results = search(memory_db, "widgets", now=NOW)
        assert len(results) == 2
        # Same multiplier → same rank (order depends on FTS ties)
        assert abs(results[0].score - results[1].score) < 0.01

    def test_flagged_ranks_below_verified_on_same_base(self, memory_db):
        insert_memory(
            memory_db, "verified widget fact",
            importance=0.8, confidence=1.0, verification="verified",
        )
        insert_memory(
            memory_db, "flagged widget fact",
            importance=0.8, confidence=1.0, verification="flagged",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) == 2
        # verified should come first
        assert "verified" in results[0].content
        assert "flagged" in results[1].content
        # flagged score should be ~0.5 × verified
        assert results[1].score < results[0].score * 0.6

    def test_contradicts_ranks_below_flagged(self, memory_db):
        insert_memory(
            memory_db, "verified widget",
            importance=0.8, confidence=1.0, verification="verified",
        )
        insert_memory(
            memory_db, "flagged widget",
            importance=0.8, confidence=1.0, verification="flagged",
        )
        insert_memory(
            memory_db, "contradicts widget",
            importance=0.8, confidence=1.0, verification="contradicts",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) == 3
        # Ordered: verified > flagged > contradicts
        assert "verified" in results[0].content
        assert "flagged" in results[1].content
        assert "contradicts" in results[2].content

    def test_breakdown_records_verification(self, memory_db):
        insert_memory(
            memory_db, "widget",
            importance=0.8, confidence=1.0, verification="flagged",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) == 1
        assert results[0].breakdown.verification == "flagged"
        assert results[0].breakdown.verification_multiplier == 0.5
        assert results[0].breakdown.trust_factor == 0.5
        assert results[0].verification == "flagged"

    def test_confidence_factor_also_demotes(self, memory_db):
        """Low-confidence memory ranks below high-confidence memory."""
        insert_memory(
            memory_db, "high confidence widget",
            importance=0.8, confidence=1.0, verification="verified",
        )
        insert_memory(
            memory_db, "low confidence widget",
            importance=0.8, confidence=0.3, verification="verified",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) == 2
        assert "high" in results[0].content
        assert "low" in results[1].content
        # Low confidence score should be ~0.3 × high confidence
        assert results[1].score < results[0].score * 0.4

    def test_trust_factor_combines_confidence_and_verification(self, memory_db):
        """A low-confidence, flagged memory gets hit by both multipliers."""
        insert_memory(
            memory_db, "widget",
            importance=0.8, confidence=0.5, verification="flagged",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) == 1
        # trust_factor = 0.5 * 0.5 = 0.25
        assert results[0].breakdown.trust_factor == pytest.approx(0.25)
        assert results[0].breakdown.confidence == 0.5
        assert results[0].breakdown.verification_multiplier == 0.5

    def test_verified_high_relevance_outranks_low_relevance(self, memory_db):
        """Sanity: trust weighting doesn't flip the normal order when
        relevance differences are large enough."""
        # Very relevant but flagged
        insert_memory(
            memory_db, "flagged precise widget match",
            importance=0.9, confidence=1.0, verification="flagged",
        )
        # Barely relevant and verified
        insert_memory(
            memory_db, "verified unrelated thing",
            importance=0.1, confidence=1.0, verification="verified",
        )
        results = search(memory_db, "widget", now=NOW)
        assert len(results) >= 1
        # Flagged but highly relevant still wins over unrelated verified
        assert "widget" in results[0].content

    def test_zero_confidence_rank_zero(self, memory_db):
        """Confidence 0 zeroes the trust factor → score 0."""
        insert_memory(
            memory_db, "widget",
            importance=1.0, confidence=0.0, verification="verified",
        )
        results = search(memory_db, "widget", now=NOW)
        # May still appear but score is 0
        assert len(results) == 1
        assert results[0].score == 0.0

    def test_unknown_verification_treated_as_default(self, memory_db):
        """Defensive: a bogus status string is caught by the CHECK
        constraint at INSERT time. But if it slipped through somehow
        (migration bug, raw SQL), the weight function returns the
        default (1.0). Verify by calling _verification_weight directly."""
        from plugins.memory.claudia.hybrid_search import (
            DEFAULT_VERIFICATION_WEIGHT,
            _verification_weight,
        )

        assert _verification_weight("bogus") == DEFAULT_VERIFICATION_WEIGHT
        assert _verification_weight(None) == DEFAULT_VERIFICATION_WEIGHT
        assert _verification_weight("") == DEFAULT_VERIFICATION_WEIGHT

    def test_verification_weights_constants(self):
        """Tripwire: lock the multiplier values."""
        from plugins.memory.claudia.hybrid_search import VERIFICATION_WEIGHTS

        assert VERIFICATION_WEIGHTS["verified"] == 1.0
        assert VERIFICATION_WEIGHTS["pending"] == 1.0
        assert VERIFICATION_WEIGHTS["flagged"] == 0.5
        assert VERIFICATION_WEIGHTS["contradicts"] == 0.3
        # All four status values must be present
        assert set(VERIFICATION_WEIGHTS.keys()) == {
            "verified", "pending", "flagged", "contradicts"
        }
