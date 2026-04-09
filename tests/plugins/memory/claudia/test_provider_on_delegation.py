"""Tests for on_delegation lifecycle hook (Phase 2C.16).

The ABC's on_delegation hook is called on the PARENT provider
when a subagent completes. Claudia captures this as a memory
row so the parent agent can recall what it delegated and what
came back.
"""

from __future__ import annotations

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


def _list_delegation_memories(p):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        rows = conn.execute(
            """
            SELECT id, content, origin, source_type, source_ref
            FROM memories
            WHERE source_type = 'delegation' AND deleted_at IS NULL
            """
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Happy path ────────────────────────────────────────────────────────


class TestOnDelegation:
    def test_creates_memory_row(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_delegation(
                task="summarize Q4 metrics",
                result="Q4 revenue up 22%, churn down 3%",
                child_session_id="subagent-7",
            )
            rows = _list_delegation_memories(p)
            assert len(rows) == 1
            row = rows[0]
            assert "summarize Q4 metrics" in row["content"]
            assert "Q4 revenue up 22%" in row["content"]
            assert row["origin"] == "inferred"
            assert row["source_type"] == "delegation"
            assert row["source_ref"] == "subagent-7"
        finally:
            p.shutdown()

    def test_multiple_delegations_accumulate(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_delegation(
                task="task 1", result="result 1",
                child_session_id="s1",
            )
            p.on_delegation(
                task="task 2", result="result 2",
                child_session_id="s2",
            )
            rows = _list_delegation_memories(p)
            assert len(rows) == 2
            refs = {r["source_ref"] for r in rows}
            assert refs == {"s1", "s2"}
        finally:
            p.shutdown()

    def test_empty_task_and_result_is_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.on_delegation(
                task="", result="", child_session_id="s1"
            )
            p.on_delegation(
                task="   ", result="   ", child_session_id="s2"
            )
            rows = _list_delegation_memories(p)
            assert rows == []
        finally:
            p.shutdown()

    def test_task_only_still_stores(self, tmp_path):
        """Delegation with a task but empty result still stores.

        The subagent may have failed, but the PARENT's observation
        is still useful: 'I tried to delegate X, got nothing back'.
        """
        p = _provider(tmp_path)
        try:
            p.on_delegation(
                task="send the email", result="",
                child_session_id="s1",
            )
            rows = _list_delegation_memories(p)
            assert len(rows) == 1
            assert "send the email" in rows[0]["content"]
        finally:
            p.shutdown()


# ─── Agent context filtering ───────────────────────────────────────────


class TestAgentContextFilter:
    @pytest.mark.parametrize("context", ["cron", "subagent", "flush"])
    def test_non_primary_skips(self, tmp_path, context):
        p = _provider(tmp_path, agent_context=context)
        try:
            p.on_delegation(
                task="x", result="y", child_session_id="s1"
            )
            rows = _list_delegation_memories(p)
            assert rows == []
        finally:
            p.shutdown()


# ─── Defensive ────────────────────────────────────────────────────────


class TestDefensive:
    def test_shutdown_provider_no_op(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        # Must not crash
        p.on_delegation(
            task="x", result="y", child_session_id="s1"
        )
