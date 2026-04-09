"""Concurrency integration tests for Claudia memory (Phase 2A.3).

Synthetic load tests that exercise the writer queue and reader pool
under realistic concurrency. Verifies:

- Multiple concurrent writer threads produce the expected row count
  with no lost writes or database corruption.
- Reader threads can run prefetch-style queries in parallel with
  writers without stalling or returning corrupted data.
- Mixed workload (3 writer threads + 2 reader threads) for a short
  duration produces consistent totals.
- shutdown() during an active workload drains pending writes.
- Reader pool under saturation (more concurrent callers than pool
  size) serializes correctly with acquire timeout.

Each test runs for a bounded duration (a few seconds at most) so
the suite stays fast. The design doc calls for a 60-second run;
we run a scaled-down version here and rely on the unit tests for
fine-grained coverage of each component.

Tests use the same _TestProvider + _FakeEmbedder pattern as
test_provider.py so we don't depend on Ollama or httpx.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.embeddings import OllamaEmbedder


# ─── Test provider with a fake embedder ─────────────────────────────────


class _ConcurrencyFakeEmbedder(OllamaEmbedder):
    """Returns a stable 3-dim vector for every call. Thread-safe."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0
        self._lock = threading.Lock()

    def _call_embed(self, text):  # type: ignore[override]
        with self._lock:
            self._call_count += 1
        return [0.1, 0.2, 0.3]


class _TestProvider(ClaudiaMemoryProvider):
    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _ConcurrencyFakeEmbedder()


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def provider(tmp_path) -> _TestProvider:
    p = _TestProvider()
    p.initialize(session_id="concurrency-test", claudia_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()


def _count_rows(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]
    finally:
        conn.close()


# ─── Pure writer concurrency ────────────────────────────────────────────


class TestConcurrentWriters:
    def test_three_writer_threads_no_lost_writes(self, provider):
        """Simulate cron + gateway + interactive each inserting memories."""
        inserts_per_thread = 50

        def _writer_worker(name):
            for i in range(inserts_per_thread):
                provider.sync_turn(
                    user_content=f"{name} user turn {i}",
                    assistant_content=f"{name} assistant response {i}",
                )

        threads = [
            threading.Thread(target=_writer_worker, args=("cron",)),
            threading.Thread(target=_writer_worker, args=("gateway",)),
            threading.Thread(target=_writer_worker, args=("interactive",)),
        ]

        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        elapsed = time.monotonic() - start

        # All threads completed without blocking on each other
        assert elapsed < 30.0
        for t in threads:
            assert not t.is_alive()

        # Flush and verify total row count
        assert provider.flush(timeout=10.0)
        expected = len(threads) * inserts_per_thread
        assert _count_rows(provider._db_path) == expected

    def test_burst_write_drains_within_timeout(self, provider):
        """Rapid burst of ~200 writes should drain in a couple seconds."""
        for i in range(200):
            provider.sync_turn(f"burst {i}", "ok")

        start = time.monotonic()
        assert provider.flush(timeout=15.0)
        elapsed = time.monotonic() - start

        # Generous upper bound — on CI a 200-row drain should take
        # well under a second, but leave headroom for slow runners.
        assert elapsed < 15.0
        assert _count_rows(provider._db_path) == 200


# ─── Writer + reader concurrency ───────────────────────────────────────


class TestMixedWorkload:
    def test_writers_and_readers_run_concurrently(self, provider):
        """3 writer threads + 2 reader threads running simultaneously."""
        stop_event = threading.Event()
        errors = []
        write_count = {"n": 0}
        read_count = {"n": 0}
        count_lock = threading.Lock()

        def _writer_worker(name):
            try:
                i = 0
                while not stop_event.is_set():
                    provider.sync_turn(f"{name} content {i}", "ok")
                    with count_lock:
                        write_count["n"] += 1
                    i += 1
                    time.sleep(0.001)  # small yield
            except Exception as exc:
                errors.append(("writer", name, exc))

        def _reader_worker(name):
            try:
                while not stop_event.is_set():
                    # prefetch is a thin wrapper around router.search
                    # with reader-pool acquire — exercises the read path
                    provider.prefetch("content")
                    with count_lock:
                        read_count["n"] += 1
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(("reader", name, exc))

        writers = [
            threading.Thread(target=_writer_worker, args=("w1",)),
            threading.Thread(target=_writer_worker, args=("w2",)),
            threading.Thread(target=_writer_worker, args=("w3",)),
        ]
        readers = [
            threading.Thread(target=_reader_worker, args=("r1",)),
            threading.Thread(target=_reader_worker, args=("r2",)),
        ]

        for t in writers + readers:
            t.start()

        # Run for 2 seconds
        time.sleep(2.0)
        stop_event.set()

        for t in writers + readers:
            t.join(timeout=5.0)

        # No exceptions anywhere
        assert not errors, f"errors in workers: {errors}"

        # Both writers and readers made meaningful progress
        assert write_count["n"] > 10, (
            f"writers only produced {write_count['n']} writes"
        )
        assert read_count["n"] > 10, (
            f"readers only produced {read_count['n']} reads"
        )

        # All writes eventually visible after flush
        assert provider.flush(timeout=10.0)
        assert _count_rows(provider._db_path) == write_count["n"]


# ─── Reader pool saturation ─────────────────────────────────────────────


class TestReaderPoolSaturation:
    def test_more_readers_than_pool_size_still_correct(self, provider):
        """8 concurrent prefetch calls against a size-4 pool."""
        # Seed some data to recall
        for i in range(10):
            provider.sync_turn(f"searchable content {i}", "ok")
        assert provider.flush(timeout=5.0)

        num_readers = 8
        results = [None] * num_readers
        errors = []

        def _reader(idx):
            try:
                block = provider.prefetch("searchable")
                results[idx] = block
            except Exception as exc:
                errors.append((idx, exc))

        threads = [
            threading.Thread(target=_reader, args=(i,))
            for i in range(num_readers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"reader errors: {errors}"

        # All readers got the same non-empty block
        for block in results:
            assert block is not None
            assert "Claudia Memory" in block
            assert "searchable content" in block


# ─── Shutdown during active workload ───────────────────────────────────


class TestShutdownDuringWorkload:
    def test_shutdown_drains_inflight_writes(self, tmp_path):
        """Writes enqueued just before shutdown must still commit."""
        p = _TestProvider()
        p.initialize(
            session_id="s1",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        db_path = p._db_path

        # Enqueue many writes from multiple threads
        num_threads = 4
        inserts_per_thread = 25

        def _worker(name):
            for i in range(inserts_per_thread):
                p.sync_turn(f"{name}-{i}", "ok")

        threads = [
            threading.Thread(target=_worker, args=(f"t{i}",))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Shutdown immediately — should drain pending writes
        p.shutdown()

        # All writes committed
        assert _count_rows(db_path) == num_threads * inserts_per_thread


# ─── Memory.remember concurrency ────────────────────────────────────────


class TestConcurrentRemember:
    def test_parallel_memory_remember_calls(self, provider):
        """memory.remember blocks on its own marker, so multiple
        concurrent callers should each get back their own unique id."""
        import json

        num_callers = 10
        returned_ids = [None] * num_callers
        errors = []

        def _caller(idx):
            try:
                result = provider.handle_tool_call(
                    "memory.remember",
                    {"content": f"concurrent remember {idx}"},
                )
                data = json.loads(result)
                returned_ids[idx] = data.get("memory_id")
            except Exception as exc:
                errors.append((idx, exc))

        threads = [
            threading.Thread(target=_caller, args=(i,))
            for i in range(num_callers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert not errors, f"caller errors: {errors}"

        # All callers got back a valid, unique id
        assert all(mid is not None for mid in returned_ids)
        assert len(set(returned_ids)) == num_callers
