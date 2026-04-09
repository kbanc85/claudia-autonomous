"""Tests for memory.correct_memory tool (Phase 2C.9).

The correction flow:
1. User: "actually Sarah's role is CTO, not VP"
2. Claude calls memory.correct_memory(id=<old memory id>,
                                      new_content="Sarah's role is CTO")
3. Provider creates a new memory row with:
     - content = the new version
     - origin = 'corrected'
     - corrected_from = <old memory id>
     - verification = 'verified' (user correction is authoritative)
     - confidence = 1.0
4. Provider marks the old memory as 'contradicts'
5. Returns the new memory in the result

This preserves an audit trail: the old memory is still in the
DB (soft signal via 'contradicts' status + corrected_from link),
and the new memory references it so future recalls know which
supersedes which.
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


def _seed_memory(p, content, *, confidence=0.7, verification="pending"):
    profile = p._profile

    def _job(conn):
        cur = conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                verification, source_type, source_ref, profile,
                created_at, accessed_at
            ) VALUES (?, 'extracted', ?, 0.5, 0, ?,
                      'conversation', 'test', ?, ?, ?)
            """,
            (
                content,
                confidence,
                verification,
                profile,
                "2026-04-09T12:00:00+00:00",
                "2026-04-09T12:00:00+00:00",
            ),
        )
        return int(cur.lastrowid)

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


def _get_raw_memory(p, mid):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        row = conn.execute(
            """
            SELECT id, content, origin, confidence, verification, corrected_from
            FROM memories WHERE id = ?
            """,
            (mid,),
        ).fetchone()
    return dict(row) if row else None


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.correct_memory" in names
        finally:
            p.shutdown()

    def test_schema_requires_id_and_new_content(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.correct_memory"
            )
            props = schema["parameters"]["properties"]
            assert "id" in props
            assert "new_content" in props
            required = schema["parameters"]["required"]
            assert "id" in required
            assert "new_content" in required
        finally:
            p.shutdown()

    def test_correct_memory_present(self, tmp_path):
        """Focused check: correct_memory is exposed. The total tool
        count lives in test_provider.py as the single source of truth."""
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.correct_memory" in names
        finally:
            p.shutdown()


# ─── Correction flow ───────────────────────────────────────────────────


class TestCorrectionFlow:
    def test_creates_new_memory_row_with_corrected_from(self, tmp_path):
        p = _provider(tmp_path)
        try:
            old_id = _seed_memory(p, "Sarah is VP Engineering")

            result_json = p.handle_tool_call(
                "memory.correct_memory",
                {
                    "id": old_id,
                    "new_content": "Sarah is CTO",
                },
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert "memory" in result

            new_memory = result["memory"]
            new_id = new_memory["id"]
            assert new_id != old_id
            assert new_memory["content"] == "Sarah is CTO"
            assert new_memory["origin"] == "corrected"
            assert new_memory["corrected_from"] == old_id
            assert new_memory["verification"] == "verified"
            assert new_memory["confidence"] == 1.0
        finally:
            p.shutdown()

    def test_old_memory_marked_as_contradicts(self, tmp_path):
        p = _provider(tmp_path)
        try:
            old_id = _seed_memory(p, "Sarah is VP Engineering")

            p.handle_tool_call(
                "memory.correct_memory",
                {
                    "id": old_id,
                    "new_content": "Sarah is CTO",
                },
            )

            # Old memory's verification status should now be
            # 'contradicts' — the new memory contradicts it.
            old_row = _get_raw_memory(p, old_id)
            assert old_row["verification"] == "contradicts"
        finally:
            p.shutdown()

    def test_old_memory_content_unchanged(self, tmp_path):
        """Correction preserves the old row's content (audit trail)."""
        p = _provider(tmp_path)
        try:
            old_id = _seed_memory(p, "Sarah is VP Engineering")
            p.handle_tool_call(
                "memory.correct_memory",
                {"id": old_id, "new_content": "Sarah is CTO"},
            )
            old_row = _get_raw_memory(p, old_id)
            assert old_row["content"] == "Sarah is VP Engineering"
        finally:
            p.shutdown()


# ─── Validation errors ─────────────────────────────────────────────────


class TestValidationErrors:
    def test_missing_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.correct_memory", {"new_content": "X"}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_missing_new_content(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.correct_memory", {"id": 1}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_empty_new_content(self, tmp_path):
        p = _provider(tmp_path)
        try:
            old_id = _seed_memory(p, "original")
            result = json.loads(p.handle_tool_call(
                "memory.correct_memory",
                {"id": old_id, "new_content": "   "},
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_unknown_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.correct_memory",
                {"id": 99999, "new_content": "X"},
            ))
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cannot_correct_other_profile_memory(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            old_id = _seed_memory(p_a, "a fact")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call(
                "memory.correct_memory",
                {"id": old_id, "new_content": "b fact"},
            ))
            assert "error" in result
        finally:
            p_b.shutdown()
