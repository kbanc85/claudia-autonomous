"""Tests for memory verification state tools (Phase 2C.8).

Three new LLM-facing tools:
- memory.verify_memory   — mark a memory as verified
- memory.flag_memory     — mark a memory as flagged (needs review)
- memory.contradicts_memory — mark a memory as contradicts
  (known conflicting with another memory)

All three take a single ``id`` parameter and return JSON with
``{ok: true, memory: {...}}`` on success or ``{error: "..."}``.
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


def _seed_memory(p, content, *, verification="pending"):
    """Seed a memory row via the writer queue and return its id."""
    profile = p._profile

    def _job(conn):
        cur = conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                verification, source_type, source_ref, profile,
                created_at, accessed_at
            ) VALUES (?, 'extracted', 0.7, 0.5, 0, ?,
                      'conversation', 'test', ?, ?, ?)
            """,
            (
                content,
                verification,
                profile,
                "2026-04-09T12:00:00+00:00",
                "2026-04-09T12:00:00+00:00",
            ),
        )
        return int(cur.lastrowid)

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Schemas ────────────────────────────────────────────────────────────


class TestSchemas:
    def test_all_three_tools_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.verify_memory" in names
            assert "memory.flag_memory" in names
            assert "memory.contradicts_memory" in names
        finally:
            p.shutdown()

    def test_total_tool_count(self, tmp_path):
        """Previously 7. 2C.8 adds 3 → 10."""
        p = _provider(tmp_path)
        try:
            assert len(p.get_tool_schemas()) == 10
        finally:
            p.shutdown()


# ─── memory.verify_memory ──────────────────────────────────────────────


class TestVerifyMemory:
    def test_verifies_pending_memory(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact")
            result_json = p.handle_tool_call(
                "memory.verify_memory", {"id": mid}
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert result["memory"]["verification"] == "verified"
        finally:
            p.shutdown()

    def test_missing_id_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call("memory.verify_memory", {})
            assert "error" in json.loads(result_json)
        finally:
            p.shutdown()

    def test_unknown_id_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.verify_memory", {"id": 99999}
            )
            assert "error" in json.loads(result_json)
        finally:
            p.shutdown()


# ─── memory.flag_memory ────────────────────────────────────────────────


class TestFlagMemory:
    def test_flags_pending_memory(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact")
            result_json = p.handle_tool_call(
                "memory.flag_memory", {"id": mid}
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert result["memory"]["verification"] == "flagged"
        finally:
            p.shutdown()

    def test_flags_already_flagged_memory(self, tmp_path):
        """Idempotent: flagging a flagged memory still succeeds."""
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact", verification="flagged")
            result_json = p.handle_tool_call(
                "memory.flag_memory", {"id": mid}
            )
            result = json.loads(result_json)
            assert result["ok"] is True
            assert result["memory"]["verification"] == "flagged"
        finally:
            p.shutdown()


# ─── memory.contradicts_memory ─────────────────────────────────────────


class TestContradictsMemory:
    def test_marks_verified_as_contradicts(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact", verification="verified")
            result_json = p.handle_tool_call(
                "memory.contradicts_memory", {"id": mid}
            )
            result = json.loads(result_json)
            assert result["ok"] is True
            assert result["memory"]["verification"] == "contradicts"
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cannot_verify_other_profile_memory(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            mid = _seed_memory(p_a, "a fact")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result_json = p_b.handle_tool_call(
                "memory.verify_memory", {"id": mid}
            )
            assert "error" in json.loads(result_json)
        finally:
            p_b.shutdown()


# ─── Dispatch regression ──────────────────────────────────────────────


class TestDispatchRegression:
    def test_existing_tools_still_work(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # memory.remember still dispatches
            result = p.handle_tool_call(
                "memory.remember", {"content": "regression sanity"}
            )
            assert "remembered" in result or "memory_id" in result
        finally:
            p.shutdown()
