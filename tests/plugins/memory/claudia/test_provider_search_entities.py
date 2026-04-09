"""Tests for memory.search_entities tool (Phase 2C.11).

Fuzzy substring search over entities. Complements memory.about
(exact single-entity lookup) for cases where the user asks
"who do I know at Acme" or "list people named Sarah".
"""

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


def _seed_entity(p, name, *, kind="person", aliases=None, importance=0.5):
    profile = p._profile

    def _job(conn):
        return entities_module.create_entity(
            conn, kind, name,
            aliases=aliases, importance=importance, profile=profile,
        ).id

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.search_entities" in names
        finally:
            p.shutdown()

    def test_schema_has_required_and_optional_params(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.search_entities"
            )
            props = schema["parameters"]["properties"]
            assert "query" in props
            assert "kind" in props
            assert "limit" in props
            assert schema["parameters"]["required"] == ["query"]
        finally:
            p.shutdown()


# ─── Happy paths ───────────────────────────────────────────────────────


class TestSearch:
    def test_substring_match(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah Chen")
            _seed_entity(p, "Sarah Kim")
            _seed_entity(p, "Bob")

            result_json = p.handle_tool_call(
                "memory.search_entities", {"query": "sarah"}
            )
            result = json.loads(result_json)
            assert "entities" in result
            names = [e["name"] for e in result["entities"]]
            assert "Sarah Chen" in names
            assert "Sarah Kim" in names
            assert "Bob" not in names
        finally:
            p.shutdown()

    def test_kind_filter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah Chen", kind="person")
            _seed_entity(p, "Acme", kind="organization")

            result = json.loads(p.handle_tool_call(
                "memory.search_entities",
                {"query": "*", "kind": "person"},
            ))
            kinds = {e["kind"] for e in result["entities"]}
            assert kinds == {"person"}
        finally:
            p.shutdown()

    def test_limit_honored(self, tmp_path):
        p = _provider(tmp_path)
        try:
            for i in range(10):
                _seed_entity(p, f"Person {i}")

            result = json.loads(p.handle_tool_call(
                "memory.search_entities",
                {"query": "person", "limit": 3},
            ))
            assert len(result["entities"]) == 3
        finally:
            p.shutdown()

    def test_wildcard_returns_all(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "A")
            _seed_entity(p, "B")
            _seed_entity(p, "C")

            result = json.loads(p.handle_tool_call(
                "memory.search_entities", {"query": "*"}
            ))
            assert len(result["entities"]) == 3
        finally:
            p.shutdown()

    def test_includes_entity_fields(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(
                p, "Sarah Chen",
                aliases=["schen"], importance=0.8,
            )
            result = json.loads(p.handle_tool_call(
                "memory.search_entities", {"query": "sarah"}
            ))
            assert len(result["entities"]) == 1
            ent = result["entities"][0]
            assert ent["name"] == "Sarah Chen"
            assert ent["kind"] == "person"
            assert ent["importance"] == 0.8
            assert "schen" in ent["aliases"]
            assert "id" in ent
        finally:
            p.shutdown()


# ─── Error and edge cases ──────────────────────────────────────────────


class TestSearchEdges:
    def test_missing_query(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.search_entities", {}
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_empty_query_returns_empty(self, tmp_path):
        """Empty string query returns [] per entities.search_entities contract."""
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            result = json.loads(p.handle_tool_call(
                "memory.search_entities", {"query": ""}
            ))
            assert result["entities"] == []
        finally:
            p.shutdown()

    def test_no_match_returns_empty(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            result = json.loads(p.handle_tool_call(
                "memory.search_entities", {"query": "xyz"}
            ))
            assert result["entities"] == []
        finally:
            p.shutdown()

    def test_invalid_kind_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.search_entities",
                {"query": "*", "kind": "bogus"},
            ))
            assert "error" in result
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_other_profile_entities_invisible(self, tmp_path):
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            _seed_entity(p_a, "user_a entity")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            result = json.loads(p_b.handle_tool_call(
                "memory.search_entities", {"query": "*"}
            ))
            # user_b has no entities of its own; user_a's are not
            # visible across profile boundary.
            assert result["entities"] == []
        finally:
            p_b.shutdown()
