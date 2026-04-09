"""Integration tests for ClaudiaMemoryProvider cost governance (Phase 2B.5).

Verifies that on_turn_start updates the provider's budget state
and that subsequent hooks (sync_turn, prefetch) degrade gracefully
when the remaining-token budget is tight.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.budget import (
    CRITICAL_TOKEN_THRESHOLD,
    DEFAULT_PREFETCH_LIMIT,
    LOW_PREFETCH_LIMIT,
    LOW_TOKEN_THRESHOLD,
)
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


class _CountingExtractor(LLMExtractor):
    def __init__(self):
        self.call_count = 0

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        return [
            ExtractedEntity(
                name="X", kind="person",
                canonical_name="x", confidence=0.9,
            )
        ]


class _CountingDetector(CommitmentDetector):
    def __init__(self):
        self.call_count = 0

    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        return [
            DetectedCommitment(
                content="do x", commitment_type="explicit", confidence=0.9
            )
        ]


class _TestProvider(ClaudiaMemoryProvider):
    def __init__(self, extractor, detector):
        super().__init__()
        self._injected_extractor = extractor
        self._injected_detector = detector

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._injected_extractor

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return self._injected_detector


def _provider(tmp_path, extractor=None, detector=None):
    extractor = extractor or _CountingExtractor()
    detector = detector or _CountingDetector()
    p = _TestProvider(extractor, detector)
    p.initialize(
        session_id="test-session",
        claudia_home=str(tmp_path),
        platform="cli",
    )
    return p, extractor, detector


# ─── Tests ──────────────────────────────────────────────────────────────


class TestOnTurnStartUpdatesState:
    def test_reads_remaining_tokens_kwarg(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            p.on_turn_start(1, "hello", remaining_tokens=4200)
            assert p._budget_state.remaining_tokens == 4200
            assert p._budget_state.turn_number == 1
        finally:
            p.shutdown()

    def test_missing_kwarg_leaves_state_unknown(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            p.on_turn_start(1, "hello")  # no remaining_tokens kwarg
            assert p._budget_state.remaining_tokens is None
        finally:
            p.shutdown()

    def test_invalid_kwarg_resets_to_none(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            p.on_turn_start(1, "hello", remaining_tokens="lots")
            assert p._budget_state.remaining_tokens is None
        finally:
            p.shutdown()


class TestSyncTurnSkipsOnCritical:
    def test_critical_budget_skips_cognitive_work(self, tmp_path):
        p, extractor, detector = _provider(tmp_path)
        try:
            # Signal critical budget
            p.on_turn_start(
                1, "hi", remaining_tokens=CRITICAL_TOKEN_THRESHOLD - 1
            )
            p.sync_turn("I'll send the proposal", "ok")
            assert p.flush(timeout=5.0)
            # Both cognitive pipelines should have been skipped
            assert extractor.call_count == 0
            assert detector.call_count == 0
        finally:
            p.shutdown()

    def test_low_budget_still_runs_cognitive(self, tmp_path):
        p, extractor, detector = _provider(tmp_path)
        try:
            p.on_turn_start(
                1, "hi", remaining_tokens=CRITICAL_TOKEN_THRESHOLD + 1
            )
            p.sync_turn("I'll send the proposal", "ok")
            assert p.flush(timeout=5.0)
            # Cognitive still runs — low ≠ critical
            assert extractor.call_count == 1
            assert detector.call_count == 1
        finally:
            p.shutdown()

    def test_normal_budget_runs_everything(self, tmp_path):
        p, extractor, detector = _provider(tmp_path)
        try:
            p.on_turn_start(1, "hi", remaining_tokens=50_000)
            p.sync_turn("I'll send the proposal", "ok")
            assert p.flush(timeout=5.0)
            assert extractor.call_count == 1
            assert detector.call_count == 1
        finally:
            p.shutdown()

    def test_memory_row_still_stored_on_critical(self, tmp_path):
        """Cost-governance skipping does NOT skip memory inserts.

        The memory row is the source of truth for the conversation
        turn. Extraction/detection are best-effort; insertion is not.
        """
        p, _, _ = _provider(tmp_path)
        try:
            p.on_turn_start(
                1, "hi", remaining_tokens=CRITICAL_TOKEN_THRESHOLD - 1
            )
            p.sync_turn("hello", "world")
            assert p.flush(timeout=5.0)
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM memories"
                ).fetchone()
            assert row["n"] == 1
        finally:
            p.shutdown()


class TestPrefetchLimitDegrades:
    def _seed_many_memories(self, p, count: int):
        """Seed more memories than DEFAULT_PREFETCH_LIMIT so we can
        observe the limit-capping behavior."""
        def _seed(conn):
            import struct
            now_iso = "2026-04-09T12:00:00+00:00"
            blob = struct.pack("<3f", 0.1, 0.2, 0.3)
            for i in range(count):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        embedding, embedding_dim, source_type, source_ref,
                        profile, created_at, accessed_at
                    ) VALUES (?, 'extracted', 0.7, ?, 0, ?, 3,
                              'conversation', 'test', 'default', ?, ?)
                    """,
                    (f"memory {i} about widgets", 0.5 + i * 0.01, blob, now_iso, now_iso),
                )

        p._writer.enqueue_and_wait(_seed, timeout=5.0)
        assert p.flush(timeout=5.0)

    def test_normal_budget_returns_default_limit(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            self._seed_many_memories(p, DEFAULT_PREFETCH_LIMIT + 5)
            p.on_turn_start(1, "widgets", remaining_tokens=50_000)
            text = p.prefetch("widgets")
            # One header line + one per result
            count = text.count("\n- ")
            assert count <= DEFAULT_PREFETCH_LIMIT
            # And strictly more than the low-budget limit
            assert count > LOW_PREFETCH_LIMIT
        finally:
            p.shutdown()

    def test_low_budget_reduces_limit(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            self._seed_many_memories(p, DEFAULT_PREFETCH_LIMIT + 5)
            p.on_turn_start(
                1, "widgets",
                remaining_tokens=CRITICAL_TOKEN_THRESHOLD + 100,
            )
            text = p.prefetch("widgets")
            count = text.count("\n- ")
            assert count <= LOW_PREFETCH_LIMIT
        finally:
            p.shutdown()

    def test_critical_budget_returns_minimal(self, tmp_path):
        p, _, _ = _provider(tmp_path)
        try:
            self._seed_many_memories(p, DEFAULT_PREFETCH_LIMIT + 5)
            p.on_turn_start(
                1, "widgets", remaining_tokens=CRITICAL_TOKEN_THRESHOLD - 1
            )
            text = p.prefetch("widgets")
            count = text.count("\n- ")
            # At critical, the limit is 3 (from CRITICAL_PREFETCH_LIMIT)
            assert count <= 3
        finally:
            p.shutdown()

    def test_prefetch_empty_result_still_empty_under_low_budget(self, tmp_path):
        """No memories in DB → prefetch returns empty regardless of budget."""
        p, _, _ = _provider(tmp_path)
        try:
            p.on_turn_start(
                1, "widgets", remaining_tokens=CRITICAL_TOKEN_THRESHOLD - 1
            )
            text = p.prefetch("widgets")
            assert text == ""
        finally:
            p.shutdown()
