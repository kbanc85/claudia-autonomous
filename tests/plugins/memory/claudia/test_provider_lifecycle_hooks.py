"""Tests for ClaudiaMemoryProvider lifecycle hooks (Phase 2C.6).

- ``on_session_end(messages)``: flush pending cognitive work at
  session boundary. No-op for non-primary agent_context and for
  a shutdown provider.
- ``on_pre_compress(messages)``: return a formatted bullet list
  of the top recent memories so the compressor can preserve
  Claudia-extracted insights even when the raw conversation is
  discarded.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
)
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import ExtractedEntity, LLMExtractor


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _CountingExtractor(LLMExtractor):
    def __init__(self):
        self.call_count = 0

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        return []


class _NoOpDetector(CommitmentDetector):
    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _TestProvider(ClaudiaMemoryProvider):
    def __init__(self, extractor: Optional[LLMExtractor] = None):
        super().__init__()
        self._injected_extractor = extractor or _CountingExtractor()

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._injected_extractor

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return _NoOpDetector()


def _provider(tmp_path, **init_kwargs) -> _TestProvider:
    p = _TestProvider()
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


def _seed_memory(p, content, *, importance=0.5):
    """Seed a memory row via the writer."""
    profile = p._profile

    def _job(conn):
        conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                source_type, source_ref, profile, created_at, accessed_at
            ) VALUES (?, 'extracted', 0.7, ?, 0, 'conversation', 'test',
                      ?, ?, ?)
            """,
            (
                content,
                importance,
                profile,
                "2026-04-09T12:00:00+00:00",
                "2026-04-09T12:00:00+00:00",
            ),
        )

    p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── on_session_end ─────────────────────────────────────────────────────


class TestOnSessionEnd:
    def test_flushes_pending_writes(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.sync_turn("hello", "world")
            # Do not explicitly flush — let on_session_end do it
            p.on_session_end([])
            # After on_session_end, the memory row should be committed
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM memories"
                ).fetchone()
            assert row["n"] == 1
        finally:
            p.shutdown()

    def test_does_not_raise_on_shutdown_provider(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        # Must not crash even though writer / reader_pool are gone
        p.on_session_end([])

    def test_empty_messages_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # Empty history → nothing to do
            p.on_session_end([])
            # Should not crash
        finally:
            p.shutdown()

    def test_non_primary_context_no_op(self, tmp_path):
        p = _provider(tmp_path, agent_context="cron")
        try:
            # sync_turn already skips for non-primary; on_session_end
            # also shouldn't do anything
            p.on_session_end([])
            # No assertion — just verifying no crash + no state change
        finally:
            p.shutdown()


# ─── on_pre_compress ────────────────────────────────────────────────────


class TestOnPreCompress:
    def test_empty_db_returns_empty_string(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = p.on_pre_compress([])
            assert result == ""
        finally:
            p.shutdown()

    def test_returns_bullet_list_of_recent_memories(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_memory(p, "fact about widgets", importance=0.8)
            _seed_memory(p, "fact about pelicans", importance=0.5)

            result = p.on_pre_compress([])
            assert "Claudia" in result  # header marker
            assert "widgets" in result
            assert "pelicans" in result
            # Bullet formatting
            assert "- " in result
        finally:
            p.shutdown()

    def test_returns_empty_on_shutdown_provider(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        # No reader pool → empty string, no crash
        assert p.on_pre_compress([]) == ""

    def test_caps_output_bullets(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # Seed many memories
            for i in range(20):
                _seed_memory(p, f"fact {i}", importance=0.5 + i * 0.01)

            result = p.on_pre_compress([])
            bullet_count = result.count("\n- ")
            # Should cap at a reasonable number (provider impl caps to 5)
            assert 0 < bullet_count <= 10
        finally:
            p.shutdown()

    def test_profile_isolation(self, tmp_path):
        """Pre-compress for one profile doesn't leak other profiles."""
        p = _provider(tmp_path, user_id="user_a")
        try:
            assert p._profile == "user_a"
            _seed_memory(p, "user_a fact", importance=0.9)

            # Seed directly into user_b profile via writer
            def _seed_other(conn):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at, accessed_at
                    ) VALUES ('user_b secret', 'extracted', 0.7, 0.9, 0,
                              'conversation', 'test', 'user_b', ?, ?)
                    """,
                    (
                        "2026-04-09T12:00:00+00:00",
                        "2026-04-09T12:00:00+00:00",
                    ),
                )

            p._writer.enqueue_and_wait(_seed_other, timeout=5.0)

            result = p.on_pre_compress([])
            # user_a fact should be present
            assert "user_a" in result
            # user_b secret MUST NOT leak into the compress summary
            assert "user_b" not in result
        finally:
            p.shutdown()
