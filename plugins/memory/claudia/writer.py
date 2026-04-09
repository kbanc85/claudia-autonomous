"""Serialized write queue for Claudia memory (Phase 2A.3).

SQLite WAL mode allows concurrent readers but still serializes
writers on a single write lock. Rather than letting multiple
caller threads contend on that lock and spin on BusyError retries,
this module routes all writes through a single dedicated
background thread.

Design:

- **One thread, one connection.** The writer thread opens its own
  ``sqlite3.Connection`` inside its own loop (so
  ``check_same_thread`` stays at the safe default). All writes to
  the memories, entities, and commitments tables go through this
  single connection.

- **Bounded job queue.** Callers enqueue ``WriterJob`` callables
  via ``enqueue()``. The queue is bounded (default 1000 jobs) so a
  runaway producer cannot exhaust memory. When the queue fills,
  ``enqueue()`` returns False and the caller decides whether to
  drop or block. The plugin is not the source of truth for
  conversation history — user turns are preserved in the session
  DB — so dropping memory writes under extreme backpressure is
  acceptable.

- **Graceful shutdown.** ``stop()`` sends a ``None`` sentinel
  through the queue *after* any currently-queued jobs. The worker
  processes every job in front of the sentinel before exiting. A
  stopped writer cannot be re-started; callers must construct a
  new instance.

- **Job isolation.** Each job is wrapped in try/except so an
  exception in one write does not kill the thread. Failed jobs
  are logged and the loop continues.

- **``flush()`` for consistency.** Tests and callers that need to
  read their own writes (e.g. ``memory.remember`` returning the
  new row id) enqueue a marker job and wait on an ``Event``. When
  the worker processes the marker, the event fires and the caller
  proceeds.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.3
concurrency design section)
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from plugins.memory.claudia import schema

logger = logging.getLogger(__name__)


#: A writer job is any callable that accepts the writer's sqlite3
#: connection and returns None (fire-and-forget) or a value (consumed
#: by ``enqueue_and_wait`` callers).
WriterJob = Callable[[sqlite3.Connection], Any]


#: Default upper bound on the queue. A gateway session with 10 turns
#: per minute fills ~600 jobs per hour, so 1000 gives an hour of
#: backpressure headroom. Adjust via the constructor if needed.
DEFAULT_MAX_QUEUE_SIZE = 1000

#: Default stop timeout — how long ``stop()`` waits for the worker
#: to drain remaining jobs before giving up.
DEFAULT_STOP_TIMEOUT_S = 10.0


class WriterQueue:
    """A single-threaded serialized writer for Claudia memory.

    Typical usage from ``provider.py``:

        writer = WriterQueue(db_path)
        writer.start()
        # ... later ...
        def _insert(conn):
            conn.execute("INSERT INTO memories ...", params)
        writer.enqueue(_insert)
        # ... on shutdown ...
        writer.stop()

    The writer is safe to share across threads: ``enqueue``,
    ``flush``, and ``stop`` can be called from any thread. Only the
    internal worker touches the underlying ``sqlite3.Connection``.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        thread_name: str = "claudia-writer",
    ) -> None:
        self._db_path = Path(db_path)
        self._max_queue_size = max_queue_size
        self._thread_name = thread_name

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop_lock = threading.Lock()
        self._stop_requested = False

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True once the worker thread has opened the DB and is consuming jobs."""
        return (
            self._ready.is_set()
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def queue_size(self) -> int:
        """Approximate number of jobs waiting in the queue."""
        return self._queue.qsize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background writer thread. Idempotent while running."""
        if self.is_running:
            return

        with self._stop_lock:
            if self._stop_requested:
                raise RuntimeError(
                    "WriterQueue has been stopped; construct a new instance"
                )

        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=self._thread_name,
        )
        self._thread.start()

        # Wait for the worker to signal it's ready (DB opened, entering
        # the consume loop). Without this, a caller that enqueues
        # immediately after start() could race the worker's setup.
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError(
                f"WriterQueue failed to start within 5s (db_path={self._db_path})"
            )

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> bool:
        """Signal the writer to drain and stop, then wait for it to join.

        Subsequent ``enqueue`` calls will fail. Previously-queued jobs
        ARE processed before the worker exits (graceful shutdown).
        Returns True if the thread joined within the timeout.
        """
        with self._stop_lock:
            if self._stop_requested:
                # Already stopping — just wait for the thread
                if self._thread is not None:
                    self._thread.join(timeout=timeout)
                return self._thread is None or not self._thread.is_alive()
            self._stop_requested = True

        # Enqueue the stop sentinel AFTER any currently-queued jobs.
        # The worker will process everything ahead of it, then exit.
        try:
            self._queue.put(None, block=True, timeout=timeout)
        except queue.Full:
            logger.warning(
                "WriterQueue stop(): queue full, worker will exit on next timeout"
            )

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    # ── Job enqueue ───────────────────────────────────────────────────

    def enqueue(
        self,
        job: WriterJob,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """Add a job to the write queue.

        By default enqueue is non-blocking: if the queue is full, the
        job is dropped and the method returns False. Set ``block=True``
        to wait for space (with optional ``timeout``).

        Returns False (without raising) when:
          - the writer has been stopped
          - the writer has not been started yet
          - the queue is full and ``block=False``
          - ``block=True`` but the timeout elapsed before space opened
        """
        with self._stop_lock:
            if self._stop_requested:
                logger.debug("enqueue() called on stopped writer; dropping job")
                return False

        if not self.is_running:
            logger.warning("enqueue() called on non-running writer; dropping job")
            return False

        try:
            self._queue.put(job, block=block, timeout=timeout)
            return True
        except queue.Full:
            logger.warning(
                "WriterQueue full (size=%d); dropping job", self._max_queue_size
            )
            return False

    def enqueue_and_wait(
        self,
        job: WriterJob,
        *,
        timeout: float = 5.0,
    ) -> Any:
        """Enqueue a job and block until the worker processes it.

        The worker invokes the job with the writer's connection. The
        job's return value is captured and returned from this method.
        Use this for writes where the caller needs the result (e.g.
        an INSERT that returns the new row id).

        Returns None on enqueue failure, queue shutdown, timeout, or
        if the job itself returned None.
        """
        result_holder: list = []
        done = threading.Event()

        def _wrapper(conn: sqlite3.Connection) -> None:
            try:
                value = job(conn)
                if value is not None:
                    result_holder.append(value)
            finally:
                done.set()

        if not self.enqueue(_wrapper, block=True, timeout=timeout):
            return None

        if not done.wait(timeout=timeout):
            logger.warning(
                "enqueue_and_wait: job did not complete within %.1fs", timeout
            )
            return None

        return result_holder[0] if result_holder else None

    def flush(self, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> bool:
        """Wait until all currently-queued jobs have been processed.

        Enqueues a marker job at the tail of the queue and waits for
        the worker to run it. Returns True if the marker was
        processed within the timeout, False on timeout or if the
        writer is not running.
        """
        if not self.is_running:
            return False

        flushed = threading.Event()

        def _marker(conn: sqlite3.Connection) -> None:
            flushed.set()

        if not self.enqueue(_marker, block=True, timeout=timeout):
            return False

        return flushed.wait(timeout=timeout)

    # ── Worker thread ─────────────────────────────────────────────────

    def _run(self) -> None:
        """Background worker main loop. Not intended for direct call."""
        try:
            conn = schema.open_connection(self._db_path)
        except Exception:
            logger.exception("Writer thread failed to open DB connection")
            return

        # Signal that we're ready. Callers blocked in start() unblock here.
        self._ready.set()

        try:
            while True:
                try:
                    job = self._queue.get(timeout=0.5)
                except queue.Empty:
                    # Periodic wakeup lets us observe stop_requested even
                    # if the queue is idle. We don't exit here unless
                    # stop was requested AND the queue is empty.
                    with self._stop_lock:
                        if self._stop_requested:
                            # A stop() call enqueues the sentinel, so
                            # normally we'd exit via the sentinel path.
                            # This branch handles a race where
                            # stop_requested was set but the sentinel
                            # hasn't been consumed yet — just keep
                            # looping until it arrives.
                            pass
                    continue

                if job is None:
                    # Sentinel received — drain any remaining jobs that
                    # were enqueued AFTER us (shouldn't be any, since
                    # stop() sets stop_requested before putting the
                    # sentinel), then exit.
                    self._queue.task_done()
                    break

                try:
                    job(conn)
                except Exception:
                    logger.exception(
                        "WriterQueue: job raised; continuing with next"
                    )
                finally:
                    self._queue.task_done()
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("Error closing writer connection")
            self._ready.clear()
