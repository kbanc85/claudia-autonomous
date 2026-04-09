"""Unit tests for plugins/memory/claudia/metrics.py (Phase 2C.17)."""

from __future__ import annotations

import threading

import pytest

from plugins.memory.claudia.metrics import Metrics


class TestBasic:
    def test_empty_registry(self):
        m = Metrics()
        assert m.snapshot() == {}
        assert m.get("anything") == 0
        assert len(m) == 0

    def test_inc_creates_counter(self):
        m = Metrics()
        m.inc("foo")
        assert m.get("foo") == 1

    def test_inc_multiple(self):
        m = Metrics()
        m.inc("foo")
        m.inc("foo")
        m.inc("foo")
        assert m.get("foo") == 3

    def test_inc_by_amount(self):
        m = Metrics()
        m.inc("foo", amount=5)
        assert m.get("foo") == 5

    def test_inc_by_zero(self):
        m = Metrics()
        m.inc("foo", amount=0)
        assert m.get("foo") == 0

    def test_inc_empty_key_is_noop(self):
        m = Metrics()
        m.inc("")
        assert m.snapshot() == {}

    def test_multiple_counters(self):
        m = Metrics()
        m.inc("a")
        m.inc("b", amount=3)
        m.inc("c", amount=7)
        snap = m.snapshot()
        assert snap == {"a": 1, "b": 3, "c": 7}


class TestSnapshot:
    def test_returns_fresh_copy(self):
        m = Metrics()
        m.inc("foo")
        snap = m.snapshot()
        snap["foo"] = 999  # mutate the copy
        # Registry is unchanged
        assert m.get("foo") == 1

    def test_sorted_keys(self):
        m = Metrics()
        m.inc("zulu")
        m.inc("alpha")
        m.inc("mike")
        keys = list(m.snapshot().keys())
        assert keys == ["alpha", "mike", "zulu"]


class TestReset:
    def test_clears_counters(self):
        m = Metrics()
        m.inc("a")
        m.inc("b")
        m.reset()
        assert m.snapshot() == {}

    def test_reset_empty_no_op(self):
        m = Metrics()
        m.reset()
        assert m.snapshot() == {}


class TestThreadSafety:
    def test_concurrent_incs(self):
        """Many threads hammering inc should not lose updates."""
        m = Metrics()
        N_THREADS = 10
        N_INCS = 1000
        barrier = threading.Barrier(N_THREADS)

        def _worker():
            barrier.wait()
            for _ in range(N_INCS):
                m.inc("counter")

        threads = [threading.Thread(target=_worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert m.get("counter") == N_THREADS * N_INCS
