"""Tests for memory.forget_memory tool (Phase 2C.14)."""

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


def _seed_memory(p, content="fact"):
    result = json.loads(p.handle_tool_call(
        "memory.remember", {"content": content}
    ))
    return int(result["memory_id"])


def _memory_is_live(p, mid):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        row = conn.execute(
            "SELECT id, deleted_at FROM memories WHERE id = ?",
            (mid,),
        ).fetchone()
    return row is not None and row["deleted_at"] is None


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.forget_memory" in names
        finally:
            p.shutdown()


# ─── Forget flow ───────────────────────────────────────────────────────


class TestForgetMemory:
    def test_soft_deletes_memory(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "bad fact")
            assert _memory_is_live(p, mid)

            result = json.loads(p.handle_tool_call(
                "memory.forget_memory", {"id": mid}
            ))
            assert result.get("ok") is True
            assert not _memory_is_live(p, mid)
        finally:
            p.shutdown()

    def test_preserves_other_memories(self, tmp_path):
        p = _provider(tmp_path)
        try:
            keep_id = _seed_memory(p, "keep this")
            forget_id = _seed_memory(p, "forget this")

            p.handle_tool_call(
                "memory.forget_memory", {"id": forget_id}
            )
            assert _memory_is_live(p, keep_id)
            assert not _memory_is_live(p, forget_id)
        finally:
            p.shutdown()

    def test_returns_forgotten_id_in_result(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact")
            result = json.loads(p.handle_tool_call(
                "memory.forget_memory", {"id": mid}
            ))
            assert result["id"] == mid
        finally:
            p.shutdown()


# ─── Errors ────────────────────────────────────────────────────────────


class TestForgetErrors:
    def test_missing_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.forget_memory", {}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_unknown_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.forget_memory", {"id": 99999}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_already_deleted_returns_error(self, tmp_path):
        """Forgetting an already-forgotten memory is an error.

        The row exists but filters out of the standard read path,
        so from the tool's perspective it doesn't exist. The handler
        reports it as unknown.
        """
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "fact")
            p.handle_tool_call(
                "memory.forget_memory", {"id": mid}
            )
            # Second forget
            result = json.loads(p.handle_tool_call(
                "memory.forget_memory", {"id": mid}
            ))
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cross_profile_blocked(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            mid = _seed_memory(p_a, "a fact")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call(
                "memory.forget_memory", {"id": mid}
            ))
            assert "error" in result
        finally:
            p_b.shutdown()
