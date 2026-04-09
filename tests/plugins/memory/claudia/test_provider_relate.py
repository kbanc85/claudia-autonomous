"""Tests for memory.relate tool (Phase 2C.10).

The memory.relate tool lets the LLM explicitly create a
directed relationship between two existing entities. It's the
operational complement to extraction: extraction finds entities
from conversation, relate connects them into a graph when the
user states a connection explicitly ("Sarah works with Bob",
"this project belongs to Acme", etc.).
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


def _seed_entity(p, name, *, kind="person"):
    profile = p._profile

    def _job(conn):
        ent = entities_module.create_entity(
            conn, kind, name, profile=profile
        )
        return ent.id

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Schema ────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.relate" in names
        finally:
            p.shutdown()

    def test_required_params(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = next(
                s for s in p.get_tool_schemas()
                if s["name"] == "memory.relate"
            )
            required = schema["parameters"]["required"]
            assert "from_name" in required
            assert "to_name" in required
            assert "type" in required
        finally:
            p.shutdown()


# ─── Happy path ────────────────────────────────────────────────────────


class TestRelateHappy:
    def test_creates_relationship(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            _seed_entity(p, "Acme")

            result_json = p.handle_tool_call(
                "memory.relate",
                {
                    "from_name": "Sarah",
                    "to_name": "Acme",
                    "type": "works_at",
                },
            )
            result = json.loads(result_json)
            assert result.get("ok") is True
            assert "relationship" in result
            rel = result["relationship"]
            assert rel["type"] == "works_at"

            # Verify the relationship exists in the DB
            assert p.flush(timeout=5.0)
            with p._reader_pool.acquire() as conn:
                sarah = entities_module.find_entity(
                    conn, "Sarah", profile="default"
                )
                rels = entities_module.get_relationships(
                    conn, sarah.id, profile="default"
                )
            assert len(rels) == 1
            assert rels[0].type == "works_at"
        finally:
            p.shutdown()

    def test_notes_preserved(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            _seed_entity(p, "Bob")
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {
                    "from_name": "Sarah",
                    "to_name": "Bob",
                    "type": "colleague",
                    "notes": "met at Q4 kickoff",
                },
            ))
            assert result["ok"] is True
            assert result["relationship"]["notes"] == "met at Q4 kickoff"
        finally:
            p.shutdown()


# ─── Error paths ───────────────────────────────────────────────────────


class TestRelateErrors:
    def test_missing_from_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {"to_name": "X", "type": "knows"},
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_missing_to_name(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {"from_name": "X", "type": "knows"},
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_missing_type(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {"from_name": "X", "to_name": "Y"},
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_from_entity_not_found(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Bob")  # only Bob exists
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {
                    "from_name": "Sarah",  # unknown
                    "to_name": "Bob",
                    "type": "knows",
                },
            ))
            assert "error" in result
            assert "sarah" in result["error"].lower()
        finally:
            p.shutdown()

    def test_to_entity_not_found(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            result = json.loads(p.handle_tool_call(
                "memory.relate",
                {
                    "from_name": "Sarah",
                    "to_name": "Bob",  # unknown
                    "type": "knows",
                },
            ))
            assert "error" in result
        finally:
            p.shutdown()

    def test_duplicate_relationship_returns_error(self, tmp_path):
        """Same (profile, from, to, type) creates IntegrityError.
        The handler catches it and returns a clean error."""
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            _seed_entity(p, "Bob")

            # First call succeeds
            first = json.loads(p.handle_tool_call(
                "memory.relate",
                {"from_name": "Sarah", "to_name": "Bob", "type": "knows"},
            ))
            assert first["ok"] is True

            # Second call hits the unique constraint
            second = json.loads(p.handle_tool_call(
                "memory.relate",
                {"from_name": "Sarah", "to_name": "Bob", "type": "knows"},
            ))
            assert "error" in second
        finally:
            p.shutdown()


# ─── Profile isolation ─────────────────────────────────────────────────


class TestProfileIsolation:
    def test_cross_profile_entity_not_visible(self, tmp_path):
        """Entities in another profile are not usable by memory.relate."""
        p_a = _provider(tmp_path, user_id="user_a")
        try:
            _seed_entity(p_a, "Sarah")
        finally:
            p_a.shutdown()

        p_b = _provider(tmp_path, user_id="user_b")
        try:
            _seed_entity(p_b, "Bob")  # only in user_b
            result = json.loads(p_b.handle_tool_call(
                "memory.relate",
                {
                    "from_name": "Sarah",  # in user_a, not user_b
                    "to_name": "Bob",
                    "type": "knows",
                },
            ))
            assert "error" in result
        finally:
            p_b.shutdown()
