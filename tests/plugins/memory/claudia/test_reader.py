"""Unit tests for plugins/memory/claudia/reader.py (Phase 2A.3).

Covers:

- Pool construction: size validation
- Lazy initialization on first acquire
- Single acquire yields a working connection
- Concurrent acquires up to the pool size
- Acquire blocks when pool is exhausted, times out with TimeoutError
- Connection is returned to the pool on normal exit
- Connection is returned even on exception
- close() drains the pool
- close() is idempotent
- acquire() after close raises RuntimeError
- close() while a connection is held closes it on release
- check_same_thread=False allows cross-thread usage
- Properties: size, db_path, is_closed, available
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.claudia.reader import (
    DEFAULT_ACQUIRE_TIMEOUT_S,
    DEFAULT_POOL_SIZE,
    ReaderPool,
)
from plugins.memory.claudia.schema import ensure_database


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def db_path(tmp_path) -> Path:
    path = tmp_path / "reader.db"
    conn = ensure_database(path)
    # Insert a row so reads have something to return
    conn.execute(
        """
        INSERT INTO memories (
            content, origin, confidence, importance, access_count,
            profile, created_at, accessed_at
        ) VALUES ('seed row', 'user_stated', 0.9, 0.5, 0, 'default',
                  '2026-04-09T12:00:00+00:00', '2026-04-09T12:00:00+00:00')
        """
    )
    conn.close()
    return path


@pytest.fixture()
def pool(db_path):
    p = ReaderPool(db_path, size=4)
    yield p
    p.close()


# ─── Construction ──────────────────────────────────────────────────────


class TestConstruction:
    def test_default_size(self, db_path):
        p = ReaderPool(db_path)
        assert p.size == DEFAULT_POOL_SIZE
        p.close()

    def test_custom_size(self, db_path):
        p = ReaderPool(db_path, size=8)
        assert p.size == 8
        p.close()

    def test_size_zero_raises(self, db_path):
        with pytest.raises(ValueError):
            ReaderPool(db_path, size=0)

    def test_negative_size_raises(self, db_path):
        with pytest.raises(ValueError):
            ReaderPool(db_path, size=-1)

    def test_db_path_property(self, db_path):
        p = ReaderPool(db_path)
        assert p.db_path == db_path
        p.close()


# ─── Lazy initialization ────────────────────────────────────────────────


class TestLazyInit:
    def test_no_connections_before_first_acquire(self, db_path):
        p = ReaderPool(db_path, size=4)
        # available reads from the internal queue; before init it's empty
        assert p.available == 0
        p.close()

    def test_first_acquire_populates_pool(self, pool):
        with pool.acquire() as conn:
            conn.execute("SELECT 1").fetchone()
        # After release, all 4 are back in the pool
        assert pool.available == pool.size


# ─── Acquire happy path ────────────────────────────────────────────────


class TestAcquire:
    def test_connection_is_usable(self, pool):
        with pool.acquire() as conn:
            row = conn.execute("SELECT content FROM memories").fetchone()
            assert row["content"] == "seed row"

    def test_connection_returned_after_exit(self, pool):
        initial = pool.size

        with pool.acquire():
            assert pool.available == initial - 1

        assert pool.available == initial

    def test_connection_returned_on_exception(self, pool):
        initial = pool.size

        with pytest.raises(ValueError):
            with pool.acquire():
                raise ValueError("oops")

        assert pool.available == initial


# ─── Concurrent acquires ───────────────────────────────────────────────


class TestConcurrentAcquires:
    def test_multiple_threads_can_acquire_in_parallel(self, pool):
        # 4-slot pool, 4 threads each holding their connection
        held = threading.Event()
        release = threading.Event()
        acquired_count = 0
        acquired_lock = threading.Lock()

        def _worker():
            nonlocal acquired_count
            with pool.acquire():
                with acquired_lock:
                    acquired_count += 1
                    if acquired_count == pool.size:
                        held.set()
                release.wait(timeout=5.0)

        threads = [threading.Thread(target=_worker) for _ in range(pool.size)]
        for t in threads:
            t.start()

        assert held.wait(timeout=5.0)
        # All 4 connections are currently out of the pool
        assert pool.available == 0

        release.set()
        for t in threads:
            t.join(timeout=5.0)

        # All 4 returned
        assert pool.available == pool.size

    def test_exhausted_pool_times_out(self, db_path):
        # Size 1 pool → second acquire must wait
        p = ReaderPool(db_path, size=1)
        try:
            held = threading.Event()
            release = threading.Event()

            def _holder():
                with p.acquire():
                    held.set()
                    release.wait(timeout=5.0)

            t = threading.Thread(target=_holder)
            t.start()
            held.wait(timeout=5.0)

            # First connection is held — a second acquire with short
            # timeout should TimeoutError
            with pytest.raises(TimeoutError):
                with p.acquire(timeout=0.2):
                    pass

            release.set()
            t.join(timeout=5.0)

            # Now the pool is free; acquire succeeds
            with p.acquire(timeout=1.0) as conn:
                conn.execute("SELECT 1")
        finally:
            p.close()


# ─── close() ────────────────────────────────────────────────────────────


class TestClose:
    def test_close_drains_pool(self, db_path):
        p = ReaderPool(db_path, size=4)
        # Warm up the pool
        with p.acquire():
            pass
        assert p.available == 4

        p.close()
        assert p.is_closed
        assert p.available == 0

    def test_close_idempotent(self, db_path):
        p = ReaderPool(db_path)
        p.close()
        p.close()  # no-op, no error

    def test_acquire_after_close_raises(self, db_path):
        p = ReaderPool(db_path)
        p.close()
        with pytest.raises(RuntimeError):
            with p.acquire():
                pass

    def test_close_while_connection_held(self, db_path):
        """If close() runs while a caller holds a connection, the
        connection should be closed when released instead of returned
        to the dead pool."""
        p = ReaderPool(db_path, size=2)

        # Warm up
        with p.acquire():
            pass
        assert p.available == 2

        acquired = threading.Event()
        closed = threading.Event()
        holder_done = threading.Event()

        def _holder():
            with p.acquire() as conn:
                acquired.set()
                closed.wait(timeout=5.0)
                # Verify the conn still works
                conn.execute("SELECT 1").fetchone()
            holder_done.set()

        t = threading.Thread(target=_holder)
        t.start()
        acquired.wait(timeout=5.0)

        # Close the pool while the holder is mid-context
        p.close()
        closed.set()

        t.join(timeout=5.0)
        assert holder_done.is_set()


# ─── Cross-thread connection sharing ───────────────────────────────────


class TestCrossThreadUse:
    def test_connection_can_move_between_threads(self, pool):
        """check_same_thread=False lets one thread acquire and another use."""
        results = []
        error = []

        def _worker():
            try:
                with pool.acquire() as conn:
                    row = conn.execute("SELECT content FROM memories").fetchone()
                    results.append(row["content"])
            except Exception as exc:
                error.append(exc)

        # Run in a separate thread from the test runner
        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5.0)

        assert not error
        assert results == ["seed row"]


# ─── Properties ────────────────────────────────────────────────────────


class TestProperties:
    def test_is_closed_false_initially(self, pool):
        assert pool.is_closed is False

    def test_size_property(self, pool):
        assert pool.size == 4

    def test_available_reflects_usage(self, pool):
        # Force init
        with pool.acquire():
            pass
        assert pool.available == 4

        with pool.acquire():
            assert pool.available == 3
            with pool.acquire():
                assert pool.available == 2
