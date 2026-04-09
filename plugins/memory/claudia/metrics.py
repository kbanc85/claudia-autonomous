"""Internal metrics counters for the Claudia memory provider (Phase 2C.17).

A thin, thread-safe counter registry for observability. Lets the
plugin track what it's actually doing without pulling in a full
metrics system (prometheus, statsd, etc.). Counters are exposed
via ``provider.get_metrics()`` and the ``memory.metrics`` tool.

Design principles:

- **Stdlib only.** Just a dict and a lock. No external deps.
- **Thread-safe.** All mutations go through the lock.
- **Process-scoped.** Counters reset on provider reinitialization.
  No persistence across restarts — this is for in-session
  observability, not long-term trend tracking.
- **Simple API.** ``inc(key, amount=1)`` and ``snapshot()``
  cover every use case. No histograms, no gauges, no rates —
  those are out of scope.
- **Hierarchical keys.** Counter names use dot-notation
  ("sync_turn.calls", "tool.memory_recall"). No enforcement,
  but following the convention keeps snapshots readable.

Standard counter keys the provider increments:

  sync_turn.calls            — total sync_turn invocations
  sync_turn.skipped           — turns skipped (non-primary context)
  memories.inserted           — memory rows inserted
  entities.upserted           — entities upserted via extraction
  commitments.inserted        — commitment rows inserted
  cognitive.extractions_run   — extraction jobs that ran
  cognitive.detections_run    — detection jobs that ran
  cognitive.extraction_errors — extractor raised an exception
  cognitive.detection_errors  — detector raised an exception
  consolidate.runs            — consolidate() calls
  verify.runs                 — verify() calls
  purge.runs                  — purge_old_soft_deletes() calls
  tool.<name>                 — handle_tool_call dispatches, per name
  tool.errors                 — handle_tool_call failures (any tool)

Reference: autonomous/fork/plans/phase-2c-handoff.md Phase 2C.17
notes.
"""

from __future__ import annotations

import threading
from typing import Dict


class Metrics:
    """Thread-safe counter registry.

    Typical usage:

        metrics = Metrics()
        metrics.inc("sync_turn.calls")
        metrics.inc("tool.memory_recall")
        snapshot = metrics.snapshot()
        # snapshot is a plain dict; mutate freely

    The underlying storage is a dict; the lock only protects
    inc/reset/snapshot operations. The snapshot returns a fresh
    copy so callers can iterate it safely after releasing the
    internal state.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def inc(self, key: str, amount: int = 1) -> None:
        """Increment ``key`` by ``amount`` (default 1). Creates
        the counter if missing."""
        if not key:
            return
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def get(self, key: str) -> int:
        """Return the current value of ``key``, 0 if unset."""
        with self._lock:
            return self._counters.get(key, 0)

    def snapshot(self) -> Dict[str, int]:
        """Return a fresh copy of all counters as a dict.

        Caller owns the returned dict — mutations don't affect
        the registry. Keys are sorted in the returned dict for
        deterministic ordering.
        """
        with self._lock:
            return dict(sorted(self._counters.items()))

    def reset(self) -> None:
        """Clear all counters."""
        with self._lock:
            self._counters.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._counters)
