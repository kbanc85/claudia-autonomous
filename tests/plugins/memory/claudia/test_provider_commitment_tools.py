"""Tests for commitment state mutation tools (Phase 2C.7).

Two new LLM-facing tools:
- ``memory.commitment_complete`` — mark a commitment as
  ``completed`` and set ``completed_at``
- ``memory.commitment_drop`` — mark a commitment as ``dropped``
  (user explicitly abandoning it)

Both take a single ``id`` parameter and return JSON with
``{ok: true, commitment: {...}}`` on success or
``{error: "..."}`` on failure (missing id, wrong profile,
soft-deleted, unknown id).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from plugins.memory.claudia import (
    ClaudiaMemoryProvider,
    commitments as commitments_module,
)
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


def _seed_commitment(p, content, *, status="open", deadline=None):
    def _job(conn):
        c = commitments_module.create_commitment(
            conn,
            content,
            status=status,
            deadline=deadline,
            profile=p._profile,
        )
        return c.id

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Schema exposure ────────────────────────────────────────────────────


class TestSchemas:
    def test_tools_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.commitment_complete" in names
            assert "memory.commitment_drop" in names
        finally:
            p.shutdown()

    def test_total_tool_count(self, tmp_path):
        """Phase 2C.5 added 2 (commitments, status). 2C.7 adds 2
        more (complete, drop). Total should be 7 now."""
        p = _provider(tmp_path)
        try:
            assert len(p.get_tool_schemas()) == 7
        finally:
            p.shutdown()

    def test_complete_schema_shape(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.commitment_complete"
            )
            props = schema["parameters"]["properties"]
            assert "id" in props
            assert props["id"]["type"] == "integer"
            assert "id" in schema["parameters"]["required"]
        finally:
            p.shutdown()


# ─── memory.commitment_complete ─────────────────────────────────────────


class TestCommitmentComplete:
    def test_completes_open_commitment(self, tmp_path):
        p = _provider(tmp_path)
        try:
            cid = _seed_commitment(p, "send the proposal")
            result_json = p.handle_tool_call(
                "memory.commitment_complete", {"id": cid}
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert result["commitment"]["id"] == cid
            assert result["commitment"]["status"] == "completed"
            assert result["commitment"]["completed_at"] is not None
        finally:
            p.shutdown()

    def test_missing_id_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.commitment_complete", {}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()

    def test_unknown_id_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.commitment_complete", {"id": 99999}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()

    def test_invalid_id_type_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.commitment_complete", {"id": "not a number"}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()

    def test_already_completed_returns_ok(self, tmp_path):
        """Completing an already-completed commitment is a no-op success."""
        p = _provider(tmp_path)
        try:
            cid = _seed_commitment(p, "x", status="completed")
            result_json = p.handle_tool_call(
                "memory.commitment_complete", {"id": cid}
            )
            result = json.loads(result_json)
            # update_commitment_status handles this: status stays
            # 'completed', completed_at is preserved (write-once)
            assert result.get("ok") is True
            assert result["commitment"]["status"] == "completed"
        finally:
            p.shutdown()


# ─── memory.commitment_drop ─────────────────────────────────────────────


class TestCommitmentDrop:
    def test_drops_open_commitment(self, tmp_path):
        p = _provider(tmp_path)
        try:
            cid = _seed_commitment(p, "do something")
            result_json = p.handle_tool_call(
                "memory.commitment_drop", {"id": cid}
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert result["commitment"]["id"] == cid
            assert result["commitment"]["status"] == "dropped"
            # Drop does NOT set completed_at (only completion does)
            assert result["commitment"]["completed_at"] is None
        finally:
            p.shutdown()

    def test_unknown_id_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.commitment_drop", {"id": 99999}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ──────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cannot_complete_other_profile_commitment(self, tmp_path):
        """A commitment in another profile must not be completable."""
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            cid = _seed_commitment(p_a, "a fact")
        finally:
            p_a.shutdown()

        # Different provider instance with different profile
        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result_json = p_b.handle_tool_call(
                "memory.commitment_complete", {"id": cid}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p_b.shutdown()


# ─── Dispatch regression ───────────────────────────────────────────────


class TestDispatchRegression:
    def test_existing_tools_still_work(self, tmp_path):
        """Existing memory.commitments tool still works alongside new ones."""
        p = _provider(tmp_path)
        try:
            cid = _seed_commitment(p, "fact")
            # List
            list_result = json.loads(
                p.handle_tool_call("memory.commitments", {})
            )
            assert len(list_result["commitments"]) == 1
            # Complete
            complete_result = json.loads(
                p.handle_tool_call(
                    "memory.commitment_complete", {"id": cid}
                )
            )
            assert complete_result["ok"] is True
            # List again with status=completed
            list_done_result = json.loads(
                p.handle_tool_call(
                    "memory.commitments", {"status": "completed"}
                )
            )
            assert len(list_done_result["commitments"]) == 1
        finally:
            p.shutdown()
