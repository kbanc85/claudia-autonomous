"""Unit tests for plugins/memory/claudia/offline.py (Phase 2A.2e).

Covers:

- MemoryMode enum and string equality
- ModeDecision dataclass structure
- OfflineRouter.select_mode() with no embedder → FTS_IMP_REC
- select_mode() with available embedder → FULL_HYBRID
- select_mode() with unavailable embedder → FTS_IMP_REC
- force_mode override for all three modes
- Mode transition logging (via caplog)
- reset() clears cached state and forces re-probe
- set_force_mode() override toggle
- search() one-shot happy path: picks right weights, embeds query,
  runs hybrid search against real DB
- search() with no embedder uses FTS_IMP_REC weights and passes
  query_embedding=None
- search() with transient embed failure mid-query re-selects mode
  within the same call
- Full integration: router + real SQLite DB + scripted fake embedder,
  verify result ordering matches the weights in use

Uses the same _FakeEmbedder pattern as test_embeddings — subclass
``OllamaEmbedder`` and override ``_call_embed`` with a scripted list.
That keeps the real availability state machine in play so tests
exercise the sticky-unavailability semantics the router depends on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from plugins.memory.claudia.embeddings import OllamaEmbedder, pack_embedding
from plugins.memory.claudia.hybrid_search import HybridWeights
from plugins.memory.claudia.offline import (
    MemoryMode,
    ModeDecision,
    OfflineRouter,
)
from plugins.memory.claudia.schema import ensure_database


# ─── Fixtures and helpers ────────────────────────────────────────────────


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def memory_db(tmp_path):
    conn = ensure_database(tmp_path / "offline.db")
    yield conn
    conn.close()


class _FakeEmbedder(OllamaEmbedder):
    """OllamaEmbedder with ``_call_embed`` swapped for a scripted fake.

    Each script entry is either a list of floats (successful embed)
    or an Exception instance (failed embed). ``_call_embed`` consumes
    entries in order and raises on exhaustion so tests fail loudly
    when they make more calls than expected.
    """

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self._script = list(script)
        self.call_count = 0

    def _call_embed(self, text):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            raise AssertionError(
                f"_call_embed invoked with no script remaining; text={text!r}"
            )
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# A 3-dim embedding for use in tests that need a consistent query vector.
EAST_VECTOR = [1.0, 0.0, 0.0]
EAST_BLOB = pack_embedding(EAST_VECTOR)


def insert_memory(
    conn,
    content: str,
    *,
    importance: float = 0.5,
    embedding=None,
    embedding_dim=None,
    profile: str = "default",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO memories (
            content, importance, access_count, embedding, embedding_dim,
            profile, origin, confidence, created_at, accessed_at
        ) VALUES (?, ?, 0, ?, ?, ?, 'user_stated', 0.9, ?, ?)
        """,
        (
            content,
            importance,
            embedding,
            embedding_dim,
            profile,
            NOW.isoformat(),
            NOW.isoformat(),
        ),
    )
    return cur.lastrowid


# ─── MemoryMode enum ────────────────────────────────────────────────────


class TestMemoryMode:
    def test_string_equality(self):
        assert MemoryMode.FULL_HYBRID == "full_hybrid"
        assert MemoryMode.FTS_IMP_REC == "fts_imp_rec"
        assert MemoryMode.PURE_FTS == "pure_fts"

    def test_values_distinct(self):
        values = {m.value for m in MemoryMode}
        assert len(values) == 3


# ─── select_mode without embedder ───────────────────────────────────────


class TestSelectModeNoEmbedder:
    def test_defaults_to_fts_imp_rec(self):
        router = OfflineRouter()
        decision = router.select_mode()
        assert decision.mode == MemoryMode.FTS_IMP_REC
        assert decision.ollama_available is False
        assert "no embedder" in decision.reason
        # Weights match the fts_imp_rec preset
        assert decision.weights == HybridWeights.fts_imp_rec()


# ─── select_mode with embedder ──────────────────────────────────────────


class TestSelectModeWithEmbedder:
    def test_available_embedder_yields_full_hybrid(self):
        embedder = _FakeEmbedder(script=[[0.1, 0.2, 0.3]])  # probe succeeds
        router = OfflineRouter(embedder)
        decision = router.select_mode()
        assert decision.mode == MemoryMode.FULL_HYBRID
        assert decision.ollama_available is True
        assert decision.weights == HybridWeights.full_hybrid()

    def test_unavailable_embedder_yields_fts_imp_rec(self):
        embedder = _FakeEmbedder(script=[ConnectionError("no daemon")])
        router = OfflineRouter(embedder)
        decision = router.select_mode()
        assert decision.mode == MemoryMode.FTS_IMP_REC
        assert decision.ollama_available is False

    def test_second_select_uses_cached_availability(self):
        embedder = _FakeEmbedder(script=[[0.1, 0.2]])  # one probe
        router = OfflineRouter(embedder)

        first = router.select_mode()
        second = router.select_mode()

        assert first.mode == second.mode == MemoryMode.FULL_HYBRID
        # Only one _call_embed — the second select used the cache
        assert embedder.call_count == 1


# ─── force_mode ──────────────────────────────────────────────────────────


class TestForceMode:
    def test_force_pure_fts_with_available_embedder(self):
        embedder = _FakeEmbedder(script=[[0.1]])
        router = OfflineRouter(embedder, force_mode=MemoryMode.PURE_FTS)
        decision = router.select_mode()
        assert decision.mode == MemoryMode.PURE_FTS
        assert decision.weights == HybridWeights.pure_fts()
        assert "forced" in decision.reason

    def test_force_full_hybrid_with_no_embedder(self):
        router = OfflineRouter(embedder=None, force_mode=MemoryMode.FULL_HYBRID)
        decision = router.select_mode()
        assert decision.mode == MemoryMode.FULL_HYBRID
        # ollama_available is correctly reported as False even though
        # the mode was forced
        assert decision.ollama_available is False

    def test_set_force_mode_override(self):
        embedder = _FakeEmbedder(script=[[0.1]])
        router = OfflineRouter(embedder)
        # Start in full_hybrid via probe
        assert router.select_mode().mode == MemoryMode.FULL_HYBRID

        # Flip to pure_fts
        router.set_force_mode(MemoryMode.PURE_FTS)
        assert router.select_mode().mode == MemoryMode.PURE_FTS

        # Clear override, back to probe-based decision
        router.set_force_mode(None)
        assert router.select_mode().mode == MemoryMode.FULL_HYBRID


# ─── Mode transitions and logging ───────────────────────────────────────


class TestModeTransitions:
    def test_first_call_no_transition_log(self, caplog):
        embedder = _FakeEmbedder(script=[[0.1]])
        router = OfflineRouter(embedder)

        with caplog.at_level(logging.INFO, logger="plugins.memory.claudia.offline"):
            router.select_mode()

        # No transition messages on the first call (nothing to transition from)
        transition_logs = [r for r in caplog.records if "transition" in r.message]
        assert transition_logs == []

    def test_transition_logged_on_mode_change(self, caplog):
        router = OfflineRouter(embedder=None)

        with caplog.at_level(logging.INFO, logger="plugins.memory.claudia.offline"):
            router.select_mode()  # FTS_IMP_REC (no embedder)
            router.set_force_mode(MemoryMode.PURE_FTS)
            router.select_mode()  # PURE_FTS

        transition_logs = [r for r in caplog.records if "transition" in r.message]
        assert len(transition_logs) == 1
        assert "fts_imp_rec" in transition_logs[0].message
        assert "pure_fts" in transition_logs[0].message

    def test_last_mode_tracked(self):
        router = OfflineRouter(embedder=None)
        assert router.last_mode is None
        router.select_mode()
        assert router.last_mode == MemoryMode.FTS_IMP_REC


# ─── reset() ────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_last_mode(self):
        embedder = _FakeEmbedder(script=[[0.1]])
        router = OfflineRouter(embedder)
        router.select_mode()
        assert router.last_mode is not None

        router.reset()
        assert router.last_mode is None

    def test_reset_reprobes_embedder(self):
        # Script: first probe OK, second probe fails — router.reset()
        # between them should cause a fresh probe to run.
        embedder = _FakeEmbedder(
            script=[[0.1, 0.2], ConnectionError("daemon died")]
        )
        router = OfflineRouter(embedder)

        first = router.select_mode()
        assert first.mode == MemoryMode.FULL_HYBRID

        router.reset()

        second = router.select_mode()
        assert second.mode == MemoryMode.FTS_IMP_REC
        # Both probes consumed
        assert embedder.call_count == 2

    def test_reset_on_router_without_embedder(self):
        router = OfflineRouter()
        router.select_mode()
        router.reset()  # should not raise


# ─── search() one-shot ──────────────────────────────────────────────────


class TestSearchOneShot:
    def test_search_with_no_embedder(self, memory_db):
        """No embedder → FTS_IMP_REC weights, query_embedding=None."""
        insert_memory(memory_db, "important thing", importance=0.9)
        insert_memory(memory_db, "trivial thing", importance=0.1)

        router = OfflineRouter(embedder=None)
        results = router.search(memory_db, "", now=NOW)

        assert len(results) == 2
        # In FTS_IMP_REC mode, importance has the highest weight (0.5)
        # so the high-importance memory should win
        assert results[0].content == "important thing"

    def test_search_with_available_embedder(self, memory_db):
        """Available embedder → FULL_HYBRID weights, query embedded."""
        insert_memory(
            memory_db,
            "semantic match",
            embedding=EAST_BLOB,
            embedding_dim=3,
            importance=0.5,
        )
        insert_memory(memory_db, "lexical only", importance=0.5)

        # Script: one probe call, one embed-for-query call
        embedder = _FakeEmbedder(script=[EAST_VECTOR, EAST_VECTOR])
        router = OfflineRouter(embedder)

        results = router.search(memory_db, "something", now=NOW)

        assert len(results) == 2
        # The embedded memory scores higher on vector similarity
        # (50% weight) than the non-embedded one
        assert results[0].content == "semantic match"
        # Both probes consumed
        assert embedder.call_count == 2

    def test_search_with_unavailable_embedder(self, memory_db):
        """Embedder down → FTS_IMP_REC, skip embed call entirely."""
        insert_memory(memory_db, "a", importance=0.9)
        insert_memory(memory_db, "b", importance=0.1)

        embedder = _FakeEmbedder(script=[ConnectionError("down")])
        router = OfflineRouter(embedder)

        results = router.search(memory_db, "anything", now=NOW)

        assert len(results) == 2
        # Importance dominates in FTS_IMP_REC mode
        assert results[0].content == "a"
        # Only the probe happened — no embed call for the query
        assert embedder.call_count == 1


# ─── Mid-query failure and re-selection ─────────────────────────────────


class TestMidQueryFailure:
    def test_embed_fails_after_probe_succeeds(self, memory_db, caplog):
        """Probe succeeded, then embed failed — mode is re-selected."""
        insert_memory(memory_db, "high importance", importance=0.9)
        insert_memory(memory_db, "low importance", importance=0.1)

        # Script:
        #   1. Probe succeeds (for is_available())
        #   2. The real embed() call fails
        embedder = _FakeEmbedder(
            script=[[0.1, 0.2, 0.3], ConnectionError("mid-query failure")]
        )
        router = OfflineRouter(embedder)

        with caplog.at_level(logging.DEBUG, logger="plugins.memory.claudia.offline"):
            results = router.search(memory_db, "some query", now=NOW)

        assert len(results) == 2
        # After the embed failure, the router re-selected and used
        # FTS_IMP_REC weights. Importance dominates → "high importance" first.
        assert results[0].content == "high importance"

        # Verify the re-selection debug log fired
        assert any("re-selecting" in r.message.lower() for r in caplog.records)

    def test_subsequent_queries_skip_embed_after_failure(self, memory_db):
        """After mid-query failure, the next search shouldn't try embed again."""
        insert_memory(memory_db, "test", importance=0.5)

        embedder = _FakeEmbedder(
            script=[[0.1], ConnectionError("down")]
        )
        router = OfflineRouter(embedder)

        router.search(memory_db, "first query", now=NOW)
        # First query consumed both script entries (probe + failed embed)
        assert embedder.call_count == 2

        router.search(memory_db, "second query", now=NOW)
        # Second query must not add to the call count — cache knows
        # the embedder is unavailable
        assert embedder.call_count == 2


# ─── Properties and introspection ───────────────────────────────────────


class TestProperties:
    def test_embedder_property(self):
        embedder = _FakeEmbedder(script=[])
        router = OfflineRouter(embedder)
        assert router.embedder is embedder

    def test_embedder_none_when_not_configured(self):
        router = OfflineRouter()
        assert router.embedder is None

    def test_force_mode_property(self):
        router = OfflineRouter(force_mode=MemoryMode.PURE_FTS)
        assert router.force_mode == MemoryMode.PURE_FTS
        router.set_force_mode(None)
        assert router.force_mode is None


# ─── ModeDecision dataclass ─────────────────────────────────────────────


class TestModeDecision:
    def test_is_frozen(self):
        decision = ModeDecision(
            mode=MemoryMode.FULL_HYBRID,
            weights=HybridWeights.full_hybrid(),
            ollama_available=True,
            reason="test",
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            decision.mode = MemoryMode.PURE_FTS  # type: ignore[misc]
