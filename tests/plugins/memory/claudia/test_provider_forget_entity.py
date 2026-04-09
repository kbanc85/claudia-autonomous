"""Tests for memory.forget_entity tool (Phase 2C.15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.memory.claudia import (
    ClaudiaMemoryProvider,
    entities as entities_module,
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


def _seed_entity(p, name, *, kind="person"):
    profile = p._profile

    def _job(conn):
        return entities_module.create_entity(
            conn, kind, name, profile=profile
        ).id

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


def _entity_is_live(p, eid):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        ent = entities_module.get_entity(conn, eid, profile=p._profile)
    return ent is not None


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.forget_entity" in names
        finally:
            p.shutdown()

    def test_accepts_name_or_id(self, tmp_path):
        """Schema requires at least one of name/id."""
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.forget_entity"
            )
            props = schema["parameters"]["properties"]
            assert "name" in props
        finally:
            p.shutdown()


# ─── Forget flow ───────────────────────────────────────────────────────


class TestForgetEntity:
    def test_soft_deletes_by_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            eid = _seed_entity(p, "Sarah Chen")
            assert _entity_is_live(p, eid)

            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {"name": "Sarah Chen"}
            ))
            assert result.get("ok") is True
            assert not _entity_is_live(p, eid)
        finally:
            p.shutdown()

    def test_case_insensitive(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah Chen")
            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {"name": "sarah chen"}
            ))
            assert result["ok"] is True
        finally:
            p.shutdown()

    def test_returns_entity_info(self, tmp_path):
        p = _provider(tmp_path)
        try:
            eid = _seed_entity(p, "Sarah")
            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {"name": "Sarah"}
            ))
            assert result["entity"]["id"] == eid
            assert result["entity"]["name"] == "Sarah"
        finally:
            p.shutdown()

    def test_preserves_other_entities(self, tmp_path):
        p = _provider(tmp_path)
        try:
            keep_id = _seed_entity(p, "Keep")
            forget_id = _seed_entity(p, "Forget")

            p.handle_tool_call(
                "memory.forget_entity", {"name": "Forget"}
            )
            assert _entity_is_live(p, keep_id)
            assert not _entity_is_live(p, forget_id)
        finally:
            p.shutdown()


# ─── Errors ────────────────────────────────────────────────────────────


class TestForgetEntityErrors:
    def test_missing_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_unknown_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {"name": "Nobody"}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_empty_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.forget_entity", {"name": ""}
            ))
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cross_profile_blocked(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            _seed_entity(p_a, "Sarah")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call(
                "memory.forget_entity", {"name": "Sarah"}
            ))
            assert "error" in result
        finally:
            p_b.shutdown()
