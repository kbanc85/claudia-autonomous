"""Integration tests for ClaudiaMemoryProvider commitment detection wire-up
(Phase 2B.2).

Exercises the path:

    provider.sync_turn(user_content, assistant_content)
      -> enqueue commitment detection on the cognitive executor
         (shared with 2B.1 entity extraction)
      -> background worker calls CommitmentDetector.detect(user_content)
      -> each DetectedCommitment is enqueued as a commitments.create_commitment
         job on the WriterQueue
      -> provider.flush() drains the cognitive pool THEN the writer
      -> commitments table contains the detected rows

Tests lock in:

- sync_turn returns before commitment detection completes (non-blocking)
- flush waits for detection AND writer drain
- non-primary agent_context skips detection entirely (same as extraction)
- detector failures do not break sync_turn or the memory insert path
- empty detection result stores no commitments
- deadline is written as ISO 8601 when the detector parses it
- source_ref propagates from sync_turn into the stored row
- shutdown drains pending detection before writer stop
- commitment detection runs on user_content only (not assistant_content)
- multiple sync_turn calls queue cleanly through the shared executor

Separate file per pitfall #8 (don't bloat test_provider.py).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from plugins.memory.claudia import (
    ClaudiaMemoryProvider,
    commitments as commitments_module,
)
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
)
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import LLMExtractor


# ─── Test doubles ───────────────────────────────────────────────────────


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _NoOpExtractor(LLMExtractor):
    """Extraction is out of scope for these tests — inject a no-op."""

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _FakeCommitmentDetector(CommitmentDetector):
    """Scripted detector for deterministic integration tests.

    Pass ``results`` as a list of (List[DetectedCommitment] | Exception)
    to script sequential ``detect`` calls. Records every call for
    assertion.
    """

    def __init__(
        self,
        results: Optional[List] = None,
        *,
        available: bool = True,
        delay: float = 0.0,
    ):
        self._results = list(results) if results is not None else []
        self._available = available
        self._delay = delay
        self.call_count = 0
        self.last_text: Optional[str] = None
        self.last_source_ref: Optional[str] = None
        self._lock = threading.Lock()

    def is_available(self):
        return self._available

    def detect(self, text, *, source_ref=""):
        with self._lock:
            self.call_count += 1
            self.last_text = text
            self.last_source_ref = source_ref
            if self._delay:
                time.sleep(self._delay)
            if not self._results:
                return []
            item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        for c in item:
            if not c.source_ref:
                c.source_ref = source_ref
        return item


class _TestProvider(ClaudiaMemoryProvider):
    """Provider subclass that injects all three cognitive fakes."""

    def __init__(self, detector: _FakeCommitmentDetector):
        super().__init__()
        self._injected_detector = detector

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return _NoOpExtractor()

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return self._injected_detector


# ─── Helpers ────────────────────────────────────────────────────────────


def _provider(
    tmp_path,
    *,
    detector: _FakeCommitmentDetector,
    **init_kwargs,
) -> _TestProvider:
    p = _TestProvider(detector)
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


def _count_commitments(p, *, status: Optional[str] = None) -> int:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        if status is None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM commitments WHERE deleted_at IS NULL"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM commitments "
                "WHERE status = ? AND deleted_at IS NULL",
                (status,),
            ).fetchone()
    return row["n"]


def _list_commitments(p) -> List[commitments_module.Commitment]:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        return commitments_module.list_commitments(conn)


def _count_memories(p) -> int:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
    return row["n"]


# ─── Tests ──────────────────────────────────────────────────────────────


class TestSyncTurnTriggersDetection:
    def test_single_turn_runs_detector_once(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send the proposal",
                        commitment_type="explicit",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll send the proposal tomorrow", "ok")
            assert p.flush(timeout=5.0)
            assert detector.call_count == 1
        finally:
            p.shutdown()

    def test_detected_commitment_lands_in_commitments_table(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send the proposal to Sarah",
                        deadline_raw="by Friday",
                        deadline_iso="2026-04-10T00:00:00+00:00",
                        confidence=0.9,
                        commitment_type="explicit",
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn(
                "I'll send the proposal to Sarah by Friday",
                "ok",
                session_id="sess-abc",
            )
            rows = _list_commitments(p)
            assert len(rows) == 1
            row = rows[0]
            assert row.content == "send the proposal to Sarah"
            assert row.deadline == "2026-04-10T00:00:00+00:00"
            assert row.status == "open"
            assert row.source_type == "conversation"
            assert row.source_ref == "sess-abc"
            # FK fields should be NULL (entity linking is 2B.3 scope)
            assert row.owner_entity_id is None
            assert row.target_entity_id is None
        finally:
            p.shutdown()

    def test_multiple_commitments_per_turn(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send the proposal",
                        confidence=0.9,
                        commitment_type="explicit",
                    ),
                    DetectedCommitment(
                        content="schedule the kickoff",
                        confidence=0.85,
                        commitment_type="explicit",
                    ),
                    DetectedCommitment(
                        content="review the contract",
                        confidence=0.6,
                        commitment_type="implicit",
                    ),
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn(
                "I'll send the proposal, schedule the kickoff, "
                "and I need to review the contract.",
                "ok",
            )
            assert _count_commitments(p) == 3
        finally:
            p.shutdown()

    def test_detector_runs_on_user_content_only(self, tmp_path):
        """The detector should be called with user_content, not the
        full combined turn (assistant content is Claudia's response
        and should not generate commitments)."""
        detector = _FakeCommitmentDetector(results=[[]])
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("user says X", "I'll do Y for you.")
            assert p.flush(timeout=5.0)
            assert detector.call_count == 1
            # The last text passed to detect() must be user_content,
            # NOT the combined "User: X\nAssistant: Y"
            assert detector.last_text == "user says X"
            assert "Assistant:" not in (detector.last_text or "")
        finally:
            p.shutdown()

    def test_sync_turn_returns_before_detection_completes(self, tmp_path):
        """sync_turn must not wait on detection. Uses a 300ms delayed
        fake detector and asserts sync_turn returns within 200ms."""
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send the proposal",
                        confidence=0.9,
                    )
                ]
            ],
            delay=0.3,
        )
        p = _provider(tmp_path, detector=detector)
        try:
            t0 = time.monotonic()
            p.sync_turn("I'll send the proposal", "ok")
            elapsed = time.monotonic() - t0
            assert elapsed < 0.2, (
                f"sync_turn blocked for {elapsed:.3f}s; expected <0.2s "
                "(commitment detection must be non-blocking)"
            )
            assert p.flush(timeout=5.0)
            assert _count_commitments(p) == 1
        finally:
            p.shutdown()

    def test_source_ref_is_session_id(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="x",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll x", "ok", session_id="sess-xyz")
            assert p.flush(timeout=5.0)
            assert detector.last_source_ref == "sess-xyz"
            rows = _list_commitments(p)
            assert rows[0].source_ref == "sess-xyz"
        finally:
            p.shutdown()

    def test_deadline_null_when_detector_returns_no_iso(self, tmp_path):
        """If the detector couldn't parse the deadline, deadline_iso
        is None and the stored row's deadline is NULL."""
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="send it",
                        deadline_raw="end of Q4",
                        deadline_iso=None,  # unparseable
                        confidence=0.8,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll send it end of Q4", "ok")
            rows = _list_commitments(p)
            assert len(rows) == 1
            assert rows[0].deadline is None
        finally:
            p.shutdown()


class TestAgentContextFiltering:
    @pytest.mark.parametrize("context", ["cron", "subagent", "flush"])
    def test_non_primary_context_skips_detection(self, tmp_path, context):
        detector = _FakeCommitmentDetector(
            results=[
                [
                    DetectedCommitment(
                        content="x",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector, agent_context=context)
        try:
            p.sync_turn("I'll x", "ok")
            assert p.flush(timeout=5.0)
            assert detector.call_count == 0
            assert _count_memories(p) == 0
            assert _count_commitments(p) == 0
        finally:
            p.shutdown()

    def test_primary_context_runs_detection(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [DetectedCommitment(content="x", confidence=0.9)]
            ]
        )
        p = _provider(tmp_path, detector=detector, agent_context="primary")
        try:
            p.sync_turn("I'll x", "ok")
            assert p.flush(timeout=5.0)
            assert detector.call_count == 1
            assert _count_commitments(p) == 1
        finally:
            p.shutdown()


class TestDetectionFailureIsolation:
    def test_detector_raises_memory_still_stored(self, tmp_path):
        detector = _FakeCommitmentDetector(results=[RuntimeError("boom")])
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll send it", "ok")  # must not raise
            assert p.flush(timeout=5.0)
            assert _count_memories(p) == 1
            assert _count_commitments(p) == 0
        finally:
            p.shutdown()

    def test_empty_results_no_rows(self, tmp_path):
        detector = _FakeCommitmentDetector(results=[[]])
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("nothing here", "ok")
            assert p.flush(timeout=5.0)
            assert _count_memories(p) == 1
            assert _count_commitments(p) == 0
            assert detector.call_count == 1
        finally:
            p.shutdown()

    def test_unavailable_detector_called_regardless(self, tmp_path):
        """is_available=False doesn't block sync_turn's submission —
        the provider always calls detect(); individual detectors
        handle the unavailable state internally. Since the fake
        just returns [], no rows are created."""
        detector = _FakeCommitmentDetector(results=[[]], available=False)
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll send it", "ok")
            assert p.flush(timeout=5.0)
            assert _count_commitments(p) == 0
        finally:
            p.shutdown()


class TestFlushAndShutdownDrains:
    def test_flush_waits_for_detection_and_write(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [DetectedCommitment(content="x", confidence=0.9)]
            ],
            delay=0.1,
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll x", "ok")
            assert p.flush(timeout=5.0)
            assert _count_commitments(p) == 1
        finally:
            p.shutdown()

    def test_shutdown_drains_pending_detection(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [DetectedCommitment(content="x", confidence=0.9)]
            ],
            delay=0.1,
        )
        p = _provider(tmp_path, detector=detector)
        p.sync_turn("I'll x", "ok")
        p.shutdown()  # no explicit flush — shutdown must drain
        assert detector.call_count == 1

    def test_multiple_turns_queue_cleanly(self, tmp_path):
        detector = _FakeCommitmentDetector(
            results=[
                [DetectedCommitment(content="send a", confidence=0.9)],
                [DetectedCommitment(content="send b", confidence=0.9)],
                [DetectedCommitment(content="send c", confidence=0.9)],
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll send a", "ok")
            p.sync_turn("I'll send b", "ok")
            p.sync_turn("I'll send c", "ok")
            assert p.flush(timeout=5.0)
            assert _count_commitments(p) == 3
            rows = _list_commitments(p)
            contents = {r.content for r in rows}
            assert contents == {"send a", "send b", "send c"}
        finally:
            p.shutdown()


class TestExtractionAndDetectionCoexist:
    """Both 2B.1 extraction and 2B.2 detection run through the same
    cognitive pool. Verify they don't interfere with each other."""

    def test_single_turn_triggers_both_pipelines(self, tmp_path):
        """A single sync_turn should produce both pipeline calls.

        Uses the no-op extractor (so no entity rows) and a scripted
        detector. Both futures should be submitted and drained by
        flush() together.
        """
        detector = _FakeCommitmentDetector(
            results=[
                [DetectedCommitment(content="x", confidence=0.9)]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            p.sync_turn("I'll x", "ok")
            assert p.flush(timeout=5.0)
            # Commitment landed
            assert _count_commitments(p) == 1
            # Memory row landed (via the normal insert path)
            assert _count_memories(p) == 1
        finally:
            p.shutdown()
