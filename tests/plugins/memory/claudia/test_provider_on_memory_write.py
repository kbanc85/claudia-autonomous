"""Integration tests for ClaudiaMemoryProvider.on_memory_write (Phase 2C.2).

The ABC's ``on_memory_write(action, target, content)`` hook lets
external providers mirror the built-in memory tool's writes. When
the user writes to MEMORY.md or USER.md via the built-in provider,
this hook fires on every registered external provider. Claudia
uses it to keep her store in sync: anything in built-in memory
should also be recallable from Claudia.

Mirror semantics:
- action='add', target='memory' → insert memory row with
  source_type='builtin_memory', origin='user_stated'
- action='add', target='user' → insert memory row with
  source_type='builtin_user', origin='user_stated'
- action='replace' → soft-delete all previous mirrors from the
  same target, then insert the new content as a fresh row
- action='remove' → soft-delete all previous mirrors from the
  same target (no insert)

These tests verify the sync behavior end-to-end through the
writer queue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import CommitmentDetector
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import LLMExtractor


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _NoOpExtractor(LLMExtractor):
    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _NoOpDetector(CommitmentDetector):
    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _TestProvider(ClaudiaMemoryProvider):
    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return _NoOpExtractor()

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return _NoOpDetector()


def _provider(tmp_path, **init_kwargs) -> _TestProvider:
    p = _TestProvider()
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


def _list_mirrored(p, source_type):
    """Return all non-deleted memory rows with a given source_type."""
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        rows = conn.execute(
            """
            SELECT id, content, origin, source_type
            FROM memories
            WHERE source_type = ? AND deleted_at IS NULL
            ORDER BY id
            """,
            (source_type,),
        ).fetchall()
    return [dict(row) for row in rows]


def _count_all_memories(p):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NULL"
        ).fetchone()
    return row["n"]


# ─── Add action ─────────────────────────────────────────────────────────


class TestOnMemoryWriteAdd:
    def test_add_memory_creates_row(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write(
                "add", "memory", "User prefers dark mode"
            )
            assert p.flush(timeout=5.0)
            rows = _list_mirrored(p, "builtin_memory")
            assert len(rows) == 1
            assert rows[0]["content"] == "User prefers dark mode"
            assert rows[0]["origin"] == "user_stated"
            assert rows[0]["source_type"] == "builtin_memory"
        finally:
            p.shutdown()

    def test_add_user_creates_row_with_user_source_type(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write(
                "add", "user", "User is a senior engineer"
            )
            assert p.flush(timeout=5.0)
            rows = _list_mirrored(p, "builtin_user")
            assert len(rows) == 1
            assert rows[0]["source_type"] == "builtin_user"
            # Should NOT appear in 'builtin_memory' bucket
            assert _list_mirrored(p, "builtin_memory") == []
        finally:
            p.shutdown()

    def test_multiple_adds_accumulate(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "fact 1")
            p.on_memory_write("add", "memory", "fact 2")
            p.on_memory_write("add", "memory", "fact 3")
            assert p.flush(timeout=5.0)
            rows = _list_mirrored(p, "builtin_memory")
            assert len(rows) == 3
            contents = sorted(r["content"] for r in rows)
            assert contents == ["fact 1", "fact 2", "fact 3"]
        finally:
            p.shutdown()

    def test_empty_content_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "")
            p.on_memory_write("add", "memory", "   ")
            assert p.flush(timeout=5.0)
            assert _count_all_memories(p) == 0
        finally:
            p.shutdown()


# ─── Replace action ─────────────────────────────────────────────────────


class TestOnMemoryWriteReplace:
    def test_replace_soft_deletes_previous_and_inserts_new(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "old fact")
            p.on_memory_write("add", "memory", "another old fact")
            p.on_memory_write(
                "replace", "memory", "brand new content"
            )
            assert p.flush(timeout=5.0)

            # Only the new content is live
            rows = _list_mirrored(p, "builtin_memory")
            assert len(rows) == 1
            assert rows[0]["content"] == "brand new content"
        finally:
            p.shutdown()

    def test_replace_does_not_touch_other_target(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "memory fact")
            p.on_memory_write("add", "user", "user fact")
            p.on_memory_write(
                "replace", "memory", "replaced memory"
            )
            assert p.flush(timeout=5.0)

            memory_rows = _list_mirrored(p, "builtin_memory")
            user_rows = _list_mirrored(p, "builtin_user")
            assert len(memory_rows) == 1
            assert memory_rows[0]["content"] == "replaced memory"
            # The user target is untouched
            assert len(user_rows) == 1
            assert user_rows[0]["content"] == "user fact"
        finally:
            p.shutdown()

    def test_replace_on_empty_target_just_inserts(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write(
                "replace", "memory", "first entry"
            )
            assert p.flush(timeout=5.0)
            rows = _list_mirrored(p, "builtin_memory")
            assert len(rows) == 1
            assert rows[0]["content"] == "first entry"
        finally:
            p.shutdown()


# ─── Remove action ──────────────────────────────────────────────────────


class TestOnMemoryWriteRemove:
    def test_remove_soft_deletes_all_mirrors_in_target(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "fact 1")
            p.on_memory_write("add", "memory", "fact 2")
            p.on_memory_write("remove", "memory", "")
            assert p.flush(timeout=5.0)
            assert _list_mirrored(p, "builtin_memory") == []
        finally:
            p.shutdown()

    def test_remove_does_not_touch_other_target(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "memory", "memory fact")
            p.on_memory_write("add", "user", "user fact")
            p.on_memory_write("remove", "memory", "")
            assert p.flush(timeout=5.0)

            assert _list_mirrored(p, "builtin_memory") == []
            user_rows = _list_mirrored(p, "builtin_user")
            assert len(user_rows) == 1
        finally:
            p.shutdown()

    def test_remove_does_not_touch_non_builtin_memories(self, tmp_path):
        """A regular memory (from sync_turn) must survive a built-in
        remove action."""
        p = _provider(tmp_path)
        try:
            # Seed a regular memory via the normal write path
            def _seed(conn):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at, accessed_at
                    ) VALUES ('conversation fact', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'sess-1', 'default', ?, ?)
                    """,
                    ("2026-04-09T12:00:00+00:00", "2026-04-09T12:00:00+00:00"),
                )

            p._writer.enqueue_and_wait(_seed, timeout=5.0)
            p.on_memory_write("add", "memory", "builtin fact")
            p.on_memory_write("remove", "memory", "")
            assert p.flush(timeout=5.0)

            # builtin mirrors gone, regular conversation memory survives
            assert _list_mirrored(p, "builtin_memory") == []
            assert _list_mirrored(p, "conversation") == [
                {
                    "id": 1,
                    "content": "conversation fact",
                    "origin": "extracted",
                    "source_type": "conversation",
                }
            ]
        finally:
            p.shutdown()


# ─── Unknown action / defensive ─────────────────────────────────────────


class TestOnMemoryWriteDefensive:
    def test_unknown_action_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # Unknown action string should not crash or mutate
            p.on_memory_write("delete_all", "memory", "x")
            assert p.flush(timeout=5.0)
            assert _count_all_memories(p) == 0
        finally:
            p.shutdown()

    def test_unknown_target_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_memory_write("add", "aliens", "should not land")
            assert p.flush(timeout=5.0)
            assert _count_all_memories(p) == 0
        finally:
            p.shutdown()

    def test_shutdown_provider_no_op(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        # Must not crash even though writer is gone
        p.on_memory_write("add", "memory", "after shutdown")


# ─── Profile isolation ──────────────────────────────────────────────────


class TestOnMemoryWriteProfileIsolation:
    def test_uses_provider_profile(self, tmp_path):
        p = _provider(tmp_path, user_id="user_a")
        try:
            assert p._profile == "user_a"
            p.on_memory_write("add", "memory", "user_a fact")
            assert p.flush(timeout=5.0)
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    """
                    SELECT profile FROM memories
                    WHERE source_type = 'builtin_memory'
                    """
                ).fetchone()
            assert row["profile"] == "user_a"
        finally:
            p.shutdown()
