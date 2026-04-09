"""Tests for memory.recent tool (Phase 2C.18).

Chronological view of memories, complementing the semantic
memory.recall with a "what did I see lately" view. Sorted by
accessed_at DESC by default; by='created_at' for insertion
order.
"""

from __future__ import annotations

import json
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


def _seed_with_timestamps(p, content, created_at, accessed_at):
    """Seed a memory row with explicit timestamps for ordering tests."""
    profile = p._profile

    def _job(conn):
        conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                source_type, source_ref, profile, created_at, accessed_at
            ) VALUES (?, 'extracted', 0.7, 0.5, 0, 'conversation', 'test',
                      ?, ?, ?)
            """,
            (content, profile, created_at, accessed_at),
        )

    p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.recent" in names
        finally:
            p.shutdown()

    def test_params(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.recent"
            )
            props = schema["parameters"]["properties"]
            assert "limit" in props
            assert "by" in props
            # Neither is required (both have defaults)
            assert schema["parameters"].get("required", []) == []
        finally:
            p.shutdown()


# ─── Happy path ────────────────────────────────────────────────────────


class TestRecent:
    def test_empty_db(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call("memory.recent", {}))
            assert result == {"memories": []}
        finally:
            p.shutdown()

    def test_default_sort_by_accessed_at_desc(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_with_timestamps(
                p, "oldest access",
                created_at="2026-04-09T10:00:00+00:00",
                accessed_at="2026-04-09T10:00:00+00:00",
            )
            _seed_with_timestamps(
                p, "middle access",
                created_at="2026-04-09T11:00:00+00:00",
                accessed_at="2026-04-09T11:00:00+00:00",
            )
            _seed_with_timestamps(
                p, "newest access",
                created_at="2026-04-09T12:00:00+00:00",
                accessed_at="2026-04-09T12:00:00+00:00",
            )

            result = json.loads(p.handle_tool_call("memory.recent", {}))
            mems = result["memories"]
            assert len(mems) == 3
            contents = [m["content"] for m in mems]
            assert contents == [
                "newest access",
                "middle access",
                "oldest access",
            ]
        finally:
            p.shutdown()

    def test_by_created_at(self, tmp_path):
        """by='created_at' sorts by insertion time regardless of access."""
        p = _provider(tmp_path)
        try:
            _seed_with_timestamps(
                p, "created first",
                created_at="2026-04-09T09:00:00+00:00",
                accessed_at="2026-04-09T15:00:00+00:00",  # accessed later
            )
            _seed_with_timestamps(
                p, "created second",
                created_at="2026-04-09T10:00:00+00:00",
                accessed_at="2026-04-09T14:00:00+00:00",
            )

            result = json.loads(p.handle_tool_call(
                "memory.recent", {"by": "created_at"}
            ))
            mems = result["memories"]
            contents = [m["content"] for m in mems]
            # Ordered by created_at DESC
            assert contents == ["created second", "created first"]
        finally:
            p.shutdown()

    def test_limit_honored(self, tmp_path):
        p = _provider(tmp_path)
        try:
            for i in range(10):
                _seed_with_timestamps(
                    p, f"memory {i}",
                    created_at=f"2026-04-09T{i:02d}:00:00+00:00",
                    accessed_at=f"2026-04-09T{i:02d}:00:00+00:00",
                )

            result = json.loads(p.handle_tool_call(
                "memory.recent", {"limit": 3}
            ))
            assert len(result["memories"]) == 3
        finally:
            p.shutdown()

    def test_includes_expected_fields(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_with_timestamps(
                p, "fact",
                created_at="2026-04-09T10:00:00+00:00",
                accessed_at="2026-04-09T10:00:00+00:00",
            )
            result = json.loads(p.handle_tool_call("memory.recent", {}))
            mem = result["memories"][0]
            for field in (
                "id", "content", "origin", "confidence", "verification",
                "source_type", "source_ref", "created_at", "accessed_at",
            ):
                assert field in mem
        finally:
            p.shutdown()


# ─── Error paths ───────────────────────────────────────────────────────


class TestRecentErrors:
    def test_invalid_by_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.recent", {"by": "magic"}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_excludes_soft_deleted(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_with_timestamps(
                p, "live",
                created_at="2026-04-09T12:00:00+00:00",
                accessed_at="2026-04-09T12:00:00+00:00",
            )

            def _seed_deleted(conn):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile,
                        created_at, accessed_at, deleted_at
                    ) VALUES ('deleted', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'test', 'default',
                              '2026-04-09T11:00:00+00:00',
                              '2026-04-09T11:00:00+00:00',
                              '2026-04-09T11:30:00+00:00')
                    """
                )

            p._writer.enqueue_and_wait(_seed_deleted, timeout=5.0)

            result = json.loads(p.handle_tool_call("memory.recent", {}))
            contents = [m["content"] for m in result["memories"]]
            assert "deleted" not in contents
            assert "live" in contents
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_other_profile_invisible(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            _seed_with_timestamps(
                p_a, "user_a memory",
                created_at="2026-04-09T12:00:00+00:00",
                accessed_at="2026-04-09T12:00:00+00:00",
            )
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call("memory.recent", {}))
            assert result == {"memories": []}
        finally:
            p_b.shutdown()
