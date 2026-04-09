"""Reader connection pool for Claudia memory (Phase 2A.3).

SQLite WAL mode allows concurrent readers without a lock, so this
pool exists for two reasons:

1. **Amortize connection setup.** Opening a sqlite3 connection,
   applying the WAL pragmas, setting row_factory, and configuring
   busy_timeout has a small but non-zero cost. A pool of N
   pre-warmed connections means the hot recall path never pays
   that cost.

2. **Parallel cursor isolation.** Even though WAL lets readers
   proceed independently at the SQLite level, a single Python
   ``sqlite3.Connection`` holds one cursor at a time. Two threads
   calling ``execute()`` on the same connection will stomp on
   each other. The pool gives each concurrent reader its own
   connection, so N threads can read in parallel without cursor
   interference.

Connections in the pool are created with ``check_same_thread=False``
because they travel between threads (one thread creates them via
``_ensure_initialized``, different threads acquire them via
``acquire()``). This is the only place in the claudia memory plugin
that disables ``check_same_thread`` — the writer thread uses its
own dedicated connection and keeps the safe default.

Usage:

    pool = ReaderPool(db_path, size=4)
    # ... later, from prefetch() or memory.recall handler ...
    with pool.acquire(timeout=5.0) as conn:
        results = hybrid_search.search(conn, query, ...)
    # Connection automatically returned to the pool on exit.

Lazy initialization: connections are opened on the first
``acquire()`` call, not in ``__init__``. That keeps provider init
cheap and ensures the DB file exists by the time we try to open it
(``provider.initialize()`` runs ``schema.ensure_database()`` before
constructing the pool).

Reference: docs/decisions/memory-provider-design.md (Phase 2A.3
concurrency design section)
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from plugins.memory.claudia import schema

logger = logging.getLogger(__name__)


#: Default number of reader connections. Sized for typical gateway
#: workloads: a small handful of sessions each running prefetch
#: concurrently. Larger deployments should bump this via the
#: constructor.
DEFAULT_POOL_SIZE = 4

#: Default timeout for ``acquire()``. Raised as TimeoutError if
#: no connection becomes available in time.
DEFAULT_ACQUIRE_TIMEOUT_S = 5.0


class ReaderPool:
    """Bounded pool of sqlite3 connections for concurrent reads.

    Thread-safe. ``acquire()`` can be called from any thread; the
    pool hands out connections one at a time and recovers them
    when the context manager exits.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        if size < 1:
            raise ValueError(f"ReaderPool size must be >= 1, got {size}")

        self._db_path = Path(db_path)
        self._size = size
        self._pool: queue.Queue = queue.Queue(maxsize=size)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._closed = False

    # ── Properties ────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Maximum number of concurrent readers supported by this pool."""
        return self._size

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def available(self) -> int:
        """Approximate number of connections currently free in the pool."""
        return self._pool.qsize()

    # ── Lazy initialization ───────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Open connections on first use. Thread-safe via ``_init_lock``."""
        with self._init_lock:
            if self._initialized or self._closed:
                return

            for _ in range(self._size):
                # check_same_thread=False is essential: the pool owns
                # the connection lifecycle but individual threads
                # execute queries on them.
                conn = schema.open_connection(
                    self._db_path,
                    check_same_thread=False,
                )
                self._pool.put(conn)

            self._initialized = True

    # ── Acquire / release ─────────────────────────────────────────────

    @contextmanager
    def acquire(
        self,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT_S,
    ) -> Iterator[sqlite3.Connection]:
        """Context manager: yields a pool connection, returns it on exit.

        If all connections are in use, waits up to ``timeout`` seconds.
        Raises ``TimeoutError`` on timeout or ``RuntimeError`` if the
        pool has been closed.

        If the pool is closed while the caller holds the connection,
        the connection is closed on exit instead of returned to the
        pool. That prevents a slow caller from stashing a connection
        into a dead pool and leaking it.
        """
        if self._closed:
            raise RuntimeError("ReaderPool is closed")

        self._ensure_initialized()

        try:
            conn = self._pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Could not acquire reader connection within {timeout}s "
                f"(pool size={self._size}, db={self._db_path})"
            )

        try:
            yield conn
        finally:
            if self._closed:
                # Pool was closed while we held the conn — close it
                # ourselves instead of returning it to a dead pool.
                try:
                    conn.close()
                except Exception:
                    logger.debug("Error closing post-close reader connection")
            else:
                # Return to pool for reuse.
                try:
                    self._pool.put(conn, block=False)
                except queue.Full:  # pragma: no cover - should never happen
                    # Would indicate a size tracking bug — log and close.
                    logger.warning(
                        "ReaderPool.put() full during release; closing conn"
                    )
                    try:
                        conn.close()
                    except Exception:
                        pass

    # ── Close ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close all connections in the pool.

        Idempotent. Safe to call while other threads hold acquired
        connections — those will be closed when released via the
        context manager's finally block.

        After close, ``acquire()`` raises RuntimeError.
        """
        with self._init_lock:
            if self._closed:
                return
            self._closed = True

        # Drain the queue, closing each connection that's currently
        # available. Acquired connections will be closed when their
        # holders release them.
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                logger.debug("Error closing pooled reader connection")
