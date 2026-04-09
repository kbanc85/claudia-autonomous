"""Tests for the new memory.commitments and memory.status tools
(Phase 2C.5).

These tools expand the provider's tool surface so the LLM can
list commitments and get system stats without going through
the module-level API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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


def _seed_commitment(p, content, *, status="open", deadline=None):
    from plugins.memory.claudia import commitments

    def _job(conn):
        c = commitments.create_commitment(
            conn,
            content,
            deadline=deadline,
            status=status,
            profile=p._profile,
        )
        return c.id

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


# ─── Tool schemas exposed ───────────────────────────────────────────────


class TestToolSchemasExposed:
    def test_commitments_tool_in_schemas(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schemas = p.get_tool_schemas()
            names = {s["name"] for s in schemas}
            assert "memory.commitments" in names
        finally:
            p.shutdown()

    def test_status_tool_in_schemas(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schemas = p.get_tool_schemas()
            names = {s["name"] for s in schemas}
            assert "memory.status" in names
        finally:
            p.shutdown()

    def test_all_tools_valid_openai_shape(self, tmp_path):
        p = _provider(tmp_path)
        try:
            for schema in p.get_tool_schemas():
                assert "name" in schema
                assert "description" in schema
                assert "parameters" in schema
                assert schema["parameters"]["type"] == "object"
                assert "properties" in schema["parameters"]
        finally:
            p.shutdown()

    def test_commitments_parameters_shape(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schemas = p.get_tool_schemas()
            commitments_schema = next(
                s for s in schemas if s["name"] == "memory.commitments"
            )
            props = commitments_schema["parameters"]["properties"]
            assert "status" in props
            assert "limit" in props
            # Status enum includes all statuses + "all"
            assert "all" in commitments_schema["parameters"]["properties"]["status"]["enum"]
        finally:
            p.shutdown()


# ─── memory.commitments handler ─────────────────────────────────────────


class TestMemoryCommitmentsTool:
    def test_empty_db_returns_empty_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call("memory.commitments", {})
            result = json.loads(result_json)
            assert result["commitments"] == []
        finally:
            p.shutdown()

    def test_lists_open_commitments_by_default(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_commitment(p, "open fact 1")
            _seed_commitment(p, "open fact 2")
            _seed_commitment(p, "done fact", status="completed")

            result_json = p.handle_tool_call("memory.commitments", {})
            result = json.loads(result_json)
            contents = [c["content"] for c in result["commitments"]]
            assert "open fact 1" in contents
            assert "open fact 2" in contents
            # Completed not in default list
            assert "done fact" not in contents
        finally:
            p.shutdown()

    def test_filter_by_completed_status(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_commitment(p, "open fact")
            _seed_commitment(p, "done fact", status="completed")

            result_json = p.handle_tool_call(
                "memory.commitments", {"status": "completed"}
            )
            result = json.loads(result_json)
            contents = [c["content"] for c in result["commitments"]]
            assert contents == ["done fact"]
        finally:
            p.shutdown()

    def test_all_status_returns_everything(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_commitment(p, "open fact")
            _seed_commitment(p, "done fact", status="completed")
            _seed_commitment(p, "overdue fact", status="overdue")

            result_json = p.handle_tool_call(
                "memory.commitments", {"status": "all"}
            )
            result = json.loads(result_json)
            assert len(result["commitments"]) == 3
        finally:
            p.shutdown()

    def test_limit_honored(self, tmp_path):
        p = _provider(tmp_path)
        try:
            for i in range(5):
                _seed_commitment(p, f"fact {i}")

            result_json = p.handle_tool_call(
                "memory.commitments", {"limit": 3}
            )
            result = json.loads(result_json)
            assert len(result["commitments"]) == 3
        finally:
            p.shutdown()

    def test_invalid_status_returns_error(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.commitments", {"status": "bogus"}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()

    def test_includes_deadline_and_id(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_commitment(
                p, "send proposal", deadline="2026-04-15"
            )
            result_json = p.handle_tool_call("memory.commitments", {})
            result = json.loads(result_json)
            assert len(result["commitments"]) == 1
            c = result["commitments"][0]
            assert c["content"] == "send proposal"
            assert c["deadline"] == "2026-04-15"
            assert "id" in c
            assert c["status"] == "open"
        finally:
            p.shutdown()


# ─── memory.status handler ──────────────────────────────────────────────


class TestMemoryStatusTool:
    def test_returns_counts(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call("memory.status", {})
            result = json.loads(result_json)
            assert "memories_count" in result
            assert "entities_count" in result
            assert "relationships_count" in result
            assert "commitments_count" in result
            # Fresh DB
            assert result["memories_count"] == 0
        finally:
            p.shutdown()

    def test_returns_mode_and_session(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call("memory.status", {})
            result = json.loads(result_json)
            assert "mode" in result
            assert "session_id" in result
            assert result["session_id"] == "test-session"
        finally:
            p.shutdown()

    def test_counts_reflect_seeded_data(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_commitment(p, "fact 1")
            _seed_commitment(p, "fact 2")
            result_json = p.handle_tool_call("memory.status", {})
            result = json.loads(result_json)
            assert result["commitments_count"] == 2
        finally:
            p.shutdown()


# ─── Dispatch through handle_tool_call ─────────────────────────────────


class TestDispatch:
    def test_unknown_tool_still_errors(self, tmp_path):
        """Adding new tools doesn't change the unknown-tool error path."""
        p = _provider(tmp_path)
        try:
            result_json = p.handle_tool_call(
                "memory.does_not_exist", {}
            )
            result = json.loads(result_json)
            assert "error" in result
        finally:
            p.shutdown()

    def test_existing_tools_still_work(self, tmp_path):
        """Regression: memory.recall / remember / about still dispatch."""
        p = _provider(tmp_path)
        try:
            result = p.handle_tool_call(
                "memory.remember", {"content": "sanity check"}
            )
            assert "remembered" in result or "memory_id" in result
        finally:
            p.shutdown()
