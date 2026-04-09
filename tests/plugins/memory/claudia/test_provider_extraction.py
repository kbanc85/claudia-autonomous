"""Integration tests for ClaudiaMemoryProvider extraction wire-up (Phase 2B.1).

These tests exercise the path:

    provider.sync_turn(user, assistant)
      -> enqueue extraction on dedicated background executor
      -> background executor calls LLMExtractor.extract(text)
      -> each ExtractedEntity is enqueued as an upsert job on WriterQueue
      -> provider.flush() drains extraction THEN writes
      -> entities table contains the extracted rows

Tests lock in these invariants:

- sync_turn returns immediately without waiting on extraction (non-blocking)
- flush() waits for extraction futures AND writer drain
- non-primary agent_context skips extraction entirely (parity with writes)
- extraction failure does not break sync_turn or corrupt memory storage
- shutdown drains extraction before stopping the writer
- multiple concurrent sync_turn calls serialize through the single-worker executor

Test file isolation rationale:
    test_provider.py is already ~56 tests / 555 lines. Per the Phase 2A.5
    handoff (pitfall #8), 2B concepts get their own test files rather
    than bloating test_provider.py further.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider, entities as entities_module
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import ExtractedEntity, LLMExtractor


# ─── Test doubles ───────────────────────────────────────────────────────


class _FakeEmbedder(OllamaEmbedder):
    """Stable embedder that always succeeds. Matches test_provider.py pattern."""

    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _FakeExtractor(LLMExtractor):
    """Scripted extractor that returns deterministic results per call.

    Pass ``results`` as a list of (List[ExtractedEntity] | Exception)
    to script sequential ``extract`` calls. Each call pops the next
    entry. If the script is exhausted, extract returns [] (permissive;
    the provider might flush during shutdown and call us once more).
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

    def is_available(self) -> bool:
        return self._available

    def extract(self, text: str, *, source_ref: str = "") -> List[ExtractedEntity]:
        with self._lock:
            self.call_count += 1
            self.last_text = text
            self.last_source_ref = source_ref
            if self._delay:
                time.sleep(self._delay)
            if not self._results:
                return []
            result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        # Stamp source_ref on each entity to match real extractor behavior
        for ent in result:
            if not ent.source_ref:
                ent.source_ref = source_ref
        return result


class _TestProvider(ClaudiaMemoryProvider):
    """Provider subclass that injects fake embedder and fake extractor."""

    def __init__(self, extractor: _FakeExtractor):
        super().__init__()
        self._injected_extractor = extractor

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._injected_extractor


# ─── Helpers ────────────────────────────────────────────────────────────


def _provider(tmp_path, *, extractor: _FakeExtractor, **init_kwargs) -> _TestProvider:
    """Build and initialize a _TestProvider for a test."""
    p = _TestProvider(extractor)
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


def _count_entities(p, *, kind: Optional[str] = None) -> int:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        if kind is None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM entities WHERE deleted_at IS NULL"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM entities "
                "WHERE kind = ? AND deleted_at IS NULL",
                (kind,),
            ).fetchone()
    return row["n"]


def _entity_names(p) -> List[str]:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        rows = conn.execute(
            "SELECT name FROM entities WHERE deleted_at IS NULL ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


def _count_memories(p) -> int:
    assert p.flush(timeout=5.0), "flush timed out"
    with p._reader_pool.acquire() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
    return row["n"]


# ─── Tests ──────────────────────────────────────────────────────────────


class TestSyncTurnTriggersExtraction:
    def test_single_turn_runs_extractor_once(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah Chen",
                        kind="person",
                        canonical_name="sarah chen",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("I met Sarah Chen today", "Noted")
            assert p.flush(timeout=5.0)
            assert extractor.call_count == 1
        finally:
            p.shutdown()

    def test_extracted_entity_lands_in_entities_table(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Acme Corp",
                        kind="organization",
                        canonical_name="acme corp",
                        confidence=0.8,
                        aliases=["Acme"],
                        attributes={"industry": "tech"},
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("Acme Corp signed a contract", "Good news")
            assert _count_entities(p) == 1
            with p._reader_pool.acquire() as conn:
                ent = entities_module.find_entity(conn, "Acme Corp", profile="default")
            assert ent is not None
            assert ent.kind == "organization"
            assert ent.aliases == ["Acme"]
            assert ent.attributes == {"industry": "tech"}
        finally:
            p.shutdown()

    def test_multiple_entities_per_turn(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah",
                        kind="person",
                        canonical_name="sarah",
                        confidence=0.9,
                    ),
                    ExtractedEntity(
                        name="Acme",
                        kind="organization",
                        canonical_name="acme",
                        confidence=0.9,
                    ),
                    ExtractedEntity(
                        name="Q4 Launch",
                        kind="project",
                        canonical_name="q4 launch",
                        confidence=0.8,
                    ),
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("Sarah at Acme is leading Q4 Launch", "Got it")
            assert _count_entities(p) == 3
            assert _count_entities(p, kind="person") == 1
            assert _count_entities(p, kind="organization") == 1
            assert _count_entities(p, kind="project") == 1
        finally:
            p.shutdown()

    def test_sync_turn_returns_before_extraction_completes(self, tmp_path):
        """sync_turn must return without waiting on the extraction call.

        Uses a delayed fake extractor to prove sync_turn returns before
        the background worker has finished. Without proper wire-up,
        sync_turn would block on the extraction call.
        """
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Alice",
                        kind="person",
                        canonical_name="alice",
                        confidence=0.9,
                    )
                ]
            ],
            delay=0.3,  # 300ms — slower than any reasonable non-blocking path
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            t0 = time.monotonic()
            p.sync_turn("Hi Alice", "Hello")
            elapsed = time.monotonic() - t0
            # sync_turn itself should return in well under 100ms even with
            # a 300ms extractor delay. Give it 200ms of slack for slow CI.
            assert elapsed < 0.2, (
                f"sync_turn blocked for {elapsed:.3f}s; expected <0.2s "
                "(extraction must be non-blocking)"
            )
            # Now flush should wait for the extraction AND subsequent write
            assert p.flush(timeout=5.0)
            assert extractor.call_count == 1
            assert _count_entities(p) == 1
        finally:
            p.shutdown()

    def test_source_ref_is_session_id(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Alice",
                        kind="person",
                        canonical_name="alice",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("Hi Alice", "Hello", session_id="sess-xyz")
            assert p.flush(timeout=5.0)
            # The extractor was called with the session_id as source_ref
            assert extractor.last_source_ref == "sess-xyz"
        finally:
            p.shutdown()


class TestAgentContextFiltering:
    """Non-primary agent_context must skip extraction (matches write skip)."""

    @pytest.mark.parametrize("context", ["cron", "subagent", "flush"])
    def test_non_primary_context_skips_extraction(self, tmp_path, context):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah",
                        kind="person",
                        canonical_name="sarah",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor, agent_context=context)
        try:
            p.sync_turn("Meeting with Sarah", "ok")
            assert p.flush(timeout=5.0)
            # Neither the extractor nor the write path should have run
            assert extractor.call_count == 0
            assert _count_memories(p) == 0
            assert _count_entities(p) == 0
        finally:
            p.shutdown()

    def test_primary_context_runs_extraction(self, tmp_path):
        """Sanity check: with context='primary' both paths run."""
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah",
                        kind="person",
                        canonical_name="sarah",
                        confidence=0.9,
                    )
                ]
            ]
        )
        p = _provider(tmp_path, extractor=extractor, agent_context="primary")
        try:
            p.sync_turn("Meeting with Sarah", "ok")
            assert p.flush(timeout=5.0)
            assert extractor.call_count == 1
            assert _count_memories(p) == 1
            assert _count_entities(p) == 1
        finally:
            p.shutdown()


class TestExtractionFailureIsolation:
    """Extraction is best-effort; failures must not break sync_turn."""

    def test_extractor_raises_memory_still_stored(self, tmp_path):
        extractor = _FakeExtractor(results=[RuntimeError("LLM exploded")])
        p = _provider(tmp_path, extractor=extractor)
        try:
            # Must not raise
            p.sync_turn("I met Sarah", "ok")
            assert p.flush(timeout=5.0)
            # The memory row should still be stored even though extraction failed
            assert _count_memories(p) == 1
            # And no entities should have been stored from the failed extraction
            assert _count_entities(p) == 0
        finally:
            p.shutdown()

    def test_extractor_returns_empty_no_entities_stored(self, tmp_path):
        extractor = _FakeExtractor(results=[[]])
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("Nothing relevant here", "ok")
            assert p.flush(timeout=5.0)
            assert _count_memories(p) == 1
            assert _count_entities(p) == 0
            assert extractor.call_count == 1
        finally:
            p.shutdown()

    def test_unavailable_extractor_skips_silently(self, tmp_path):
        """is_available=False is expected at init time (Ollama not running).

        The provider must still construct successfully and sync_turn
        must still work (memories stored), but the extractor's extract()
        is allowed to return [] so no entities are created.
        """
        extractor = _FakeExtractor(available=False)
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("text", "ok")
            assert p.flush(timeout=5.0)
            assert _count_memories(p) == 1
            assert _count_entities(p) == 0
        finally:
            p.shutdown()


class TestFlushAndShutdownDrains:
    def test_flush_waits_for_extraction_and_write(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah",
                        kind="person",
                        canonical_name="sarah",
                        confidence=0.9,
                    )
                ]
            ],
            delay=0.1,
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("hi Sarah", "hi")
            # Before flush, the entity might or might not be there
            # (race between sync_turn return and background work)
            # After flush, it MUST be there.
            assert p.flush(timeout=5.0)
            assert _count_entities(p) == 1
        finally:
            p.shutdown()

    def test_shutdown_drains_pending_extraction(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Sarah",
                        kind="person",
                        canonical_name="sarah",
                        confidence=0.9,
                    )
                ]
            ],
            delay=0.1,
        )
        p = _provider(tmp_path, extractor=extractor)
        p.sync_turn("hi Sarah", "hi")
        # Shutdown WITHOUT an explicit flush — shutdown itself must drain
        p.shutdown()
        assert extractor.call_count == 1, (
            "shutdown did not drain the extraction queue"
        )

    def test_multiple_turns_queue_cleanly(self, tmp_path):
        extractor = _FakeExtractor(
            results=[
                [
                    ExtractedEntity(
                        name="Alice",
                        kind="person",
                        canonical_name="alice",
                        confidence=0.9,
                    )
                ],
                [
                    ExtractedEntity(
                        name="Bob",
                        kind="person",
                        canonical_name="bob",
                        confidence=0.9,
                    )
                ],
                [
                    ExtractedEntity(
                        name="Carol",
                        kind="person",
                        canonical_name="carol",
                        confidence=0.9,
                    )
                ],
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("met Alice", "ok")
            p.sync_turn("met Bob", "ok")
            p.sync_turn("met Carol", "ok")
            assert p.flush(timeout=5.0)
            assert _count_entities(p, kind="person") == 3
            assert set(_entity_names(p)) == {"Alice", "Bob", "Carol"}
        finally:
            p.shutdown()
