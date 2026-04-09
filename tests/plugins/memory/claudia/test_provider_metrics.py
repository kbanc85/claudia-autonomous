"""Provider integration tests for metrics instrumentation (Phase 2C.17).

Verifies that the counters actually increment at the expected
code points: sync_turn, extractions, detections, consolidate,
verify, purge, and per-tool dispatches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
)
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import ExtractedEntity, LLMExtractor


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _ScriptedExtractor(LLMExtractor):
    def __init__(self, results: Optional[list] = None):
        self._results = list(results) if results is not None else []

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        if not self._results:
            return []
        return self._results.pop(0)


class _ScriptedDetector(CommitmentDetector):
    def __init__(self, results: Optional[list] = None):
        self._results = list(results) if results is not None else []

    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        if not self._results:
            return []
        return self._results.pop(0)


class _NoOpExtractor(LLMExtractor):
    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _NoOpDetector(CommitmentDetector):
    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _TestProvider(ClaudiaMemoryProvider):
    def __init__(
        self,
        extractor: Optional[LLMExtractor] = None,
        detector: Optional[CommitmentDetector] = None,
    ):
        super().__init__()
        self._injected_extractor = extractor or _NoOpExtractor()
        self._injected_detector = detector or _NoOpDetector()

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._injected_extractor

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return self._injected_detector


def _provider(tmp_path, **kwargs) -> _TestProvider:
    ext = kwargs.pop("extractor", None)
    det = kwargs.pop("detector", None)
    p = _TestProvider(extractor=ext, detector=det)
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


# ─── Metrics tool schema and handler ───────────────────────────────────


class TestMetricsTool:
    def test_tool_in_schema_list(self, tmp_path):
        p = _provider(tmp_path)
        try:
            names = {s["name"] for s in p.get_tool_schemas()}
            assert "memory.metrics" in names
        finally:
            p.shutdown()

    def test_empty_metrics_returns_at_least_one_counter(self, tmp_path):
        """Calling memory.metrics itself increments tool.memory.metrics,
        so the snapshot is never truly empty."""
        p = _provider(tmp_path)
        try:
            result = json.loads(p.handle_tool_call("memory.metrics", {}))
            assert "metrics" in result
            # The metrics call itself registers
            assert result["metrics"].get("tool.memory.metrics", 0) >= 1
        finally:
            p.shutdown()

    def test_get_metrics_public_accessor(self, tmp_path):
        p = _provider(tmp_path)
        try:
            snap = p.get_metrics()
            assert isinstance(snap, dict)
        finally:
            p.shutdown()


# ─── sync_turn instrumentation ────────────────────────────────────────


class TestSyncTurnInstrumentation:
    def test_sync_turn_calls_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.sync_turn("hi", "hello")
            p.sync_turn("again", "hi")
            snap = p.get_metrics()
            assert snap.get("sync_turn.calls") == 2
        finally:
            p.shutdown()

    def test_memories_inserted_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.sync_turn("hi", "hello")
            p.sync_turn("again", "hi")
            assert p.flush(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("memories.inserted") == 2
        finally:
            p.shutdown()

    def test_sync_turn_skipped_counter(self, tmp_path):
        p = _provider(tmp_path, agent_context="cron")
        try:
            p.sync_turn("hi", "hello")
            snap = p.get_metrics()
            assert snap.get("sync_turn.skipped") == 1
            assert snap.get("sync_turn.calls", 0) == 0
        finally:
            p.shutdown()


# ─── Cognitive pipeline instrumentation ───────────────────────────────


class TestCognitiveInstrumentation:
    def test_extraction_run_counter(self, tmp_path):
        ext = _ScriptedExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah", kind="person",
                        canonical_name="sarah", confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=ext)
        try:
            p.sync_turn("I met Sarah", "ok")
            assert p.flush(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("cognitive.extractions_run") == 1
            assert snap.get("entities.upserted") == 1
        finally:
            p.shutdown()

    def test_detection_run_counter(self, tmp_path):
        det = _ScriptedDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send proposal", confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=det)
        try:
            p.sync_turn("I'll send the proposal", "ok")
            assert p.flush(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("cognitive.detections_run") == 1
            assert snap.get("commitments.inserted") == 1
        finally:
            p.shutdown()

    def test_extraction_error_counter(self, tmp_path):
        class _BrokenExtractor(LLMExtractor):
            def extract(self, text, *, source_ref=""):
                raise RuntimeError("boom")

        p = _provider(tmp_path, extractor=_BrokenExtractor())
        try:
            p.sync_turn("text", "ok")
            assert p.flush(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("cognitive.extraction_errors") == 1
        finally:
            p.shutdown()


# ─── Maintenance instrumentation ──────────────────────────────────────


class TestMaintenanceInstrumentation:
    def test_consolidate_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.consolidate(timeout=5.0)
            p.consolidate(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("consolidate.runs") == 2
        finally:
            p.shutdown()

    def test_verify_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.verify(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("verify.runs") == 1
        finally:
            p.shutdown()

    def test_purge_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.purge_old_soft_deletes(timeout=5.0)
            snap = p.get_metrics()
            assert snap.get("purge.runs") == 1
        finally:
            p.shutdown()


# ─── Tool dispatch instrumentation ────────────────────────────────────


class TestToolDispatchInstrumentation:
    def test_per_tool_counters(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.handle_tool_call("memory.recall", {"query": "x"})
            p.handle_tool_call("memory.recall", {"query": "y"})
            p.handle_tool_call("memory.about", {"name": "nobody"})
            snap = p.get_metrics()
            assert snap.get("tool.memory.recall") == 2
            assert snap.get("tool.memory.about") == 1
        finally:
            p.shutdown()

    def test_unknown_tool_error_counter(self, tmp_path):
        p = _provider(tmp_path)
        try:
            p.handle_tool_call("memory.nonsense", {})
            snap = p.get_metrics()
            assert snap.get("tool.errors") >= 1
        finally:
            p.shutdown()
