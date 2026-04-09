"""Unit tests for plugins/memory/claudia/writer.py (Phase 2A.3).

Covers:

- WriterQueue lifecycle: start, stop, idempotency, cannot restart after stop
- Basic enqueue: job runs, result visible after flush
- Non-blocking enqueue on full queue returns False
- Blocking enqueue with timeout
- Job exception does not kill the worker thread
- stop() drains pending jobs before exiting (graceful shutdown)
- stop() is idempotent
- enqueue after stop returns False
- flush() blocks until all current jobs are processed
- enqueue_and_wait() captures the job's return value
- Concurrent enqueue from multiple threads
- Properties: is_running, queue_size, db_path

Tests use a real SQLite DB via ``schema.ensure_database`` so writes
actually commit. Each test has its own tmp_path-scoped DB so state
doesn't leak.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.claudia.schema import ensure_database
from plugins.memory.claudia.writer import (
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_STOP_TIMEOUT_S,
    WriterQueue,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def db_path(tmp_path) -> Path:
    """Initialize a DB file and return its path. The writer opens its own conn."""
    path = tmp_path / "writer.db"
    conn = ensure_database(path)
    conn.close()
    return path


@pytest.fixture()
def writer(db_path):
    """Start a WriterQueue, yield it, stop it on teardown."""
    w = WriterQueue(db_path)
    w.start()
    yield w
    w.stop(timeout=5.0)


def _count_memories(db_path: Path) -> int:
    """Helper: open a fresh conn and count memory rows."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return row[0]
    finally:
        conn.close()


def _insert_job(content: str):
    """Factory for a writer job that inserts a memory row."""

    def _job(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                profile, created_at, accessed_at
            ) VALUES (?, 'user_stated', 0.9, 0.5, 0, 'default', ?, ?)
            """,
            (content, "2026-04-09T12:00:00+00:00", "2026-04-09T12:00:00+00:00"),
        )
        return int(cur.lastrowid)

    return _job


# ─── Lifecycle ──────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_then_running(self, db_path):
        w = WriterQueue(db_path)
        assert not w.is_running
        w.start()
        try:
            assert w.is_running
        finally:
            w.stop()

    def test_start_idempotent(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        try:
            w.start()  # second call should be a no-op
            assert w.is_running
        finally:
            w.stop()

    def test_stop_joins_thread(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        assert w.stop(timeout=5.0) is True
        assert not w.is_running

    def test_stop_idempotent(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        w.stop()
        w.stop()  # second call is a no-op

    def test_cannot_restart_after_stop(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        w.stop()
        with pytest.raises(RuntimeError):
            w.start()

    def test_stop_without_start(self, db_path):
        """stop() on a never-started writer should return True (nothing to stop)."""
        w = WriterQueue(db_path)
        assert w.stop() is True


# ─── Enqueue happy path ─────────────────────────────────────────────────


class TestEnqueueBasics:
    def test_enqueue_runs_job(self, writer, db_path):
        writer.enqueue(_insert_job("hello"))
        assert writer.flush(timeout=5.0)
        assert _count_memories(db_path) == 1

    def test_multiple_enqueues(self, writer, db_path):
        for i in range(5):
            writer.enqueue(_insert_job(f"memory {i}"))
        assert writer.flush(timeout=5.0)
        assert _count_memories(db_path) == 5

    def test_enqueue_after_stop_returns_false(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        w.stop()
        assert w.enqueue(_insert_job("post-stop")) is False

    def test_enqueue_before_start_returns_false(self, db_path):
        w = WriterQueue(db_path)
        assert w.enqueue(_insert_job("pre-start")) is False


# ─── Job exceptions ─────────────────────────────────────────────────────


class TestJobExceptions:
    def test_exception_does_not_kill_worker(self, writer, db_path):
        def _bad_job(conn):
            raise RuntimeError("oops")

        writer.enqueue(_bad_job)
        writer.enqueue(_insert_job("survives"))
        assert writer.flush(timeout=5.0)
        # The bad job failed but the good job ran
        assert _count_memories(db_path) == 1
        assert writer.is_running

    def test_multiple_exceptions_handled(self, writer, db_path):
        for i in range(3):
            writer.enqueue(lambda conn, i=i: (_ for _ in ()).throw(ValueError(f"err{i}")))
        writer.enqueue(_insert_job("after errors"))
        assert writer.flush(timeout=5.0)
        assert _count_memories(db_path) == 1
        assert writer.is_running


# ─── flush() ────────────────────────────────────────────────────────────


class TestFlush:
    def test_flush_empty_queue(self, writer):
        # Flush on an empty queue should return quickly
        assert writer.flush(timeout=5.0) is True

    def test_flush_waits_for_all_queued(self, writer, db_path):
        # Enqueue multiple jobs
        for i in range(10):
            writer.enqueue(_insert_job(f"row {i}"))
        assert writer.flush(timeout=5.0)
        assert _count_memories(db_path) == 10

    def test_flush_on_stopped_writer_returns_false(self, db_path):
        w = WriterQueue(db_path)
        w.start()
        w.stop()
        assert w.flush() is False


# ─── enqueue_and_wait ──────────────────────────────────────────────────


class TestEnqueueAndWait:
    def test_captures_return_value(self, writer):
        result = writer.enqueue_and_wait(_insert_job("capture me"))
        assert isinstance(result, int)
        assert result > 0

    def test_returns_none_when_job_returns_none(self, writer):
        def _no_return(conn):
            conn.execute("SELECT 1")
            # implicit None

        result = writer.enqueue_and_wait(_no_return)
        assert result is None

    def test_timeout_returns_none(self, writer):
        def _slow(conn):
            time.sleep(2.0)

        # Give the slow job a very short timeout
        result = writer.enqueue_and_wait(_slow, timeout=0.1)
        assert result is None


# ─── Graceful shutdown ─────────────────────────────────────────────────


class TestGracefulShutdown:
    def test_stop_drains_pending_jobs(self, db_path):
        w = WriterQueue(db_path)
        w.start()

        # Enqueue several jobs and immediately stop.
        # stop() should drain them before exiting.
        for i in range(5):
            w.enqueue(_insert_job(f"drain {i}"))

        assert w.stop(timeout=5.0) is True
        assert _count_memories(db_path) == 5


# ─── Concurrent enqueue ────────────────────────────────────────────────


class TestConcurrentEnqueue:
    def test_many_threads_enqueue_safely(self, writer, db_path):
        num_threads = 10
        inserts_per_thread = 20

        def _worker(thread_id):
            for i in range(inserts_per_thread):
                writer.enqueue(_insert_job(f"t{thread_id}-{i}"))

        threads = [
            threading.Thread(target=_worker, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert writer.flush(timeout=10.0)
        assert _count_memories(db_path) == num_threads * inserts_per_thread


# ─── Backpressure ───────────────────────────────────────────────────────


class TestBackpressure:
    def test_non_blocking_enqueue_on_full_returns_false(self, db_path):
        # Use a tiny queue so we can fill it
        w = WriterQueue(db_path, max_queue_size=2)
        w.start()
        try:
            # Enqueue a slow job to stall the queue
            blocker_started = threading.Event()
            can_proceed = threading.Event()

            def _blocker(conn):
                blocker_started.set()
                can_proceed.wait(timeout=5.0)

            w.enqueue(_blocker)
            blocker_started.wait(timeout=5.0)

            # The worker is now stuck in _blocker. We can fill the queue.
            # Queue capacity is 2, and the blocker was dequeued already,
            # so the queue is currently empty but the worker is busy.
            # Put 2 more jobs → queue full.
            assert w.enqueue(_insert_job("a")) is True
            assert w.enqueue(_insert_job("b")) is True
            # Third should fail (non-blocking, queue full)
            assert w.enqueue(_insert_job("c"), block=False) is False

            can_proceed.set()  # let the blocker finish
        finally:
            w.stop(timeout=5.0)


# ─── Properties ─────────────────────────────────────────────────────────


class TestProperties:
    def test_db_path(self, db_path):
        w = WriterQueue(db_path)
        assert w.db_path == db_path

    def test_is_running_false_before_start(self, db_path):
        w = WriterQueue(db_path)
        assert w.is_running is False

    def test_queue_size_reflects_pending(self, db_path):
        w = WriterQueue(db_path, max_queue_size=10)
        w.start()
        try:
            # Block the worker
            can_proceed = threading.Event()
            started = threading.Event()

            def _blocker(conn):
                started.set()
                can_proceed.wait(timeout=5.0)

            w.enqueue(_blocker)
            started.wait(timeout=5.0)

            # Now enqueue a few jobs
            for i in range(3):
                w.enqueue(_insert_job(f"q{i}"))

            assert w.queue_size == 3

            can_proceed.set()
            assert w.flush(timeout=5.0)
        finally:
            w.stop(timeout=5.0)
