"""Tests for memory.trace tool (Phase 2C.12).

Walks a memory's correction chain backward (predecessors via
corrected_from) and forward (successors that were
corrected_from this memory). Used for provenance audits: "why
does Claudia think X?" traces back to the original extracted
memory and forward to any corrections.
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


def _seed_memory(p, content):
    """Seed via memory.remember so we go through the real insert path."""
    result_json = p.handle_tool_call(
        "memory.remember", {"content": content}
    )
    result = json.loads(result_json)
    assert "memory_id" in result
    return int(result["memory_id"])


def _correct(p, old_id, new_content):
    """Apply a correction via memory.correct_memory."""
    result_json = p.handle_tool_call(
        "memory.correct_memory",
        {"id": old_id, "new_content": new_content},
    )
    result = json.loads(result_json)
    assert result["ok"] is True
    return int(result["memory"]["id"])


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.trace" in names
        finally:
            p.shutdown()

    def test_id_required(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.trace"
            )
            assert schema["parameters"]["required"] == ["id"]
        finally:
            p.shutdown()


# ─── Simple trace ──────────────────────────────────────────────────────


class TestSimpleTrace:
    def test_uncorrected_memory_returns_single_entry(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "simple fact")
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": mid}
            ))
            assert "chain" in result
            assert len(result["chain"]) == 1
            assert result["chain"][0]["id"] == mid
        finally:
            p.shutdown()

    def test_chain_entry_has_core_fields(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _seed_memory(p, "hi")
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": mid}
            ))
            entry = result["chain"][0]
            assert "id" in entry
            assert "content" in entry
            assert "origin" in entry
            assert "verification" in entry
            assert "created_at" in entry
            assert "corrected_from" in entry
        finally:
            p.shutdown()


# ─── Correction chain ──────────────────────────────────────────────────


class TestCorrectionChain:
    def test_single_correction(self, tmp_path):
        p = _provider(tmp_path)
        try:
            v1 = _seed_memory(p, "Sarah is VP")
            v2 = _correct(p, v1, "Sarah is CTO")

            # Trace from either should show both
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": v2}
            ))
            chain = result["chain"]
            assert len(chain) == 2
            ids = [c["id"] for c in chain]
            assert v1 in ids
            assert v2 in ids

            # Trace from the old one also shows the full chain
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": v1}
            ))
            ids = [c["id"] for c in result["chain"]]
            assert v1 in ids
            assert v2 in ids
        finally:
            p.shutdown()

    def test_multi_step_correction(self, tmp_path):
        p = _provider(tmp_path)
        try:
            v1 = _seed_memory(p, "Sarah is junior")
            v2 = _correct(p, v1, "Sarah is senior")
            v3 = _correct(p, v2, "Sarah is CTO")

            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": v3}
            ))
            chain = result["chain"]
            ids_in_chain = {c["id"] for c in chain}
            assert ids_in_chain == {v1, v2, v3}
        finally:
            p.shutdown()

    def test_chain_ordered_oldest_first(self, tmp_path):
        """Chain is ordered by created_at ASC so walking the list
        matches the timeline of corrections."""
        p = _provider(tmp_path)
        try:
            v1 = _seed_memory(p, "first")
            v2 = _correct(p, v1, "second")
            v3 = _correct(p, v2, "third")

            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": v3}
            ))
            chain = result["chain"]
            # v1 should come before v2 before v3
            idx = {c["id"]: i for i, c in enumerate(chain)}
            assert idx[v1] < idx[v2] < idx[v3]
        finally:
            p.shutdown()


# ─── Error paths ──────────────────────────────────────────────────────


class TestTraceErrors:
    def test_missing_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call("memory.trace", {}))
            assert "error" in result
        finally:
            p.shutdown()

    def test_invalid_id_type(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": "abc"}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_unknown_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.trace", {"id": 99999}
            ))
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cross_profile_returns_unknown(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            mid = _seed_memory(p_a, "user_a fact")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call(
                "memory.trace", {"id": mid}
            ))
            assert "error" in result
        finally:
            p_b.shutdown()
