"""Ollama embedding pipeline for the Claudia memory provider (Phase 2A.2b).

Thin, stdlib-friendly wrapper around the Ollama HTTP API for generating
vector embeddings of memory content. Designed for the offline-capable,
write-queue-driven provider in ``provider.py`` (sub-task 2A.2f).

Design principles:

- **Lazy availability check.** The ABC in ``agent/memory_provider.py``
  explicitly forbids network calls in ``initialize()``. Probing Ollama
  on agent startup would hang the session whenever the daemon is down.
  Instead, ``OllamaEmbedder`` probes the Ollama daemon on the *first*
  call to ``embed()`` or ``is_available()`` and caches the result.

- **Silent degrade.** When Ollama is unreachable or the embedding
  model isn't loaded, ``embed()`` returns ``None``. Callers store NULL
  in the ``memories.embedding`` BLOB column, and ``hybrid_search.py``
  (2A.2c) reweights the ranking formula to drop the 50% vector term.
  This is the first of the three-tier offline degradation paths
  described in the design doc.

- **Raw bytes output.** Results are little-endian float32 bytes suitable
  for direct insertion into the BLOB column defined in ``schema.py``.
  No numpy dependency — just stdlib ``struct``. The same layout is what
  ``sqlite-vec`` expects for its virtual tables, so the stored blob is
  ready for vector similarity search when the extension is available.

- **Hardcoded host with env var override.** ``http://localhost:11434`` is
  the default. Set ``CLAUDIA_OLLAMA_HOST`` to point at a remote Ollama,
  a docker-hosted daemon, or a test stub.

- **Thread-safe state.** ``embed()`` is called from the serialized
  write-queue worker thread (Phase 2A.3), while ``is_available()`` may
  be called from the gateway session thread for system-prompt stats.
  A single mutex protects the availability cache and cached dimension.

Integration point: ``provider.py`` will construct a single
``OllamaEmbedder`` per provider instance during ``initialize()`` and
hold it for the lifetime of the session. The writer worker pulls text
off the queue and calls ``embed()`` synchronously before inserting the
memory row. Because embeddings happen on the background thread, a slow
or unreachable Ollama never blocks the user's chat turn.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.2b)
"""

from __future__ import annotations

import logging
import os
import struct
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Defaults ────────────────────────────────────────────────────────────

#: Default Ollama host URL. Override via the ``CLAUDIA_OLLAMA_HOST`` env var.
DEFAULT_HOST = "http://localhost:11434"

#: Default embedding model. Claudia v1 shipped this as the baseline; it
#: produces 384-dimensional vectors and runs comfortably on CPU. Users
#: opt into larger models via ``plugins/memory/claudia/config.yaml`` in
#: Phase 2A.2f (provider registration).
DEFAULT_MODEL = "all-minilm:l6-v2"

#: Default HTTP timeout in seconds. Kept low so a hung Ollama does not
#: stall the write worker. The ABC's sync_turn() is called after each
#: conversation turn, and five seconds is already generous for a local
#: CPU embedding model.
DEFAULT_TIMEOUT_S = 5.0

#: Environment variable that overrides ``DEFAULT_HOST``.
ENV_HOST = "CLAUDIA_OLLAMA_HOST"


# ─── BLOB encoding helpers ──────────────────────────────────────────────

def pack_embedding(vector) -> bytes:
    """Encode a float vector as little-endian float32 bytes for SQLite.

    Matches the layout that ``sqlite-vec`` virtual tables expect when
    the extension is loaded. Raises ``ValueError`` on an empty vector
    because a zero-dimension embedding is never a legitimate value
    from a working model.
    """
    if vector is None:
        raise ValueError("Cannot pack a None embedding")

    # Accept any iterable of numbers (list, tuple, array.array, etc.)
    floats = tuple(float(x) for x in vector)
    if not floats:
        raise ValueError("Cannot pack an empty embedding")

    return struct.pack(f"<{len(floats)}f", *floats)


def unpack_embedding(blob: bytes) -> list:
    """Decode a little-endian float32 BLOB back into a list of floats.

    Inverse of ``pack_embedding``. Used by hybrid_search.py (2A.2c) when
    computing cosine similarity without the sqlite-vec extension, and
    by tests that want to verify round-trip encoding. Returns an empty
    list for a None or empty blob rather than raising, since NULL
    embeddings are a legitimate "offline" state.
    """
    if not blob:
        return []

    # Length must be a multiple of 4 because each float32 is 4 bytes.
    if len(blob) % 4 != 0:
        raise ValueError(
            f"Embedding BLOB length {len(blob)} is not a multiple of 4; "
            f"blob is corrupt or was written by an incompatible encoder"
        )

    dim = len(blob) // 4
    return list(struct.unpack(f"<{dim}f", blob))


# ─── OllamaEmbedder ──────────────────────────────────────────────────────

class OllamaEmbedder:
    """Generate embeddings via a local Ollama daemon with lazy probing.

    Typical lifecycle:

        embedder = OllamaEmbedder()                   # no network I/O
        # ... later, from the write-queue worker ...
        result = embedder.embed("some memory text")   # lazy probe here
        if result is None:
            embedding_blob, embedding_dim = None, None  # offline mode
        else:
            embedding_blob, embedding_dim = result

    The first ``embed()`` call probes the Ollama daemon by issuing a
    single embedding request for a probe string. On success, the
    embedding dimension is cached in ``dimension`` and subsequent calls
    skip the probe. On failure, availability is cached as ``False`` and
    all subsequent calls return ``None`` immediately — we do not spam
    a downed daemon. The caller can call ``reset_availability()`` to
    force a re-probe (e.g. after a session boundary).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._host = (host or os.environ.get(ENV_HOST) or DEFAULT_HOST).rstrip("/")
        self._timeout = timeout

        # _available: None = unchecked, True = known available,
        #             False = known unavailable (will not re-probe)
        self._available: Optional[bool] = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

    # ── Public properties ──────────────────────────────────────────────

    @property
    def model(self) -> str:
        """The embedding model name passed to Ollama (e.g. 'all-minilm:l6-v2')."""
        return self._model

    @property
    def host(self) -> str:
        """The resolved Ollama host URL (env var override applied)."""
        return self._host

    @property
    def dimension(self) -> Optional[int]:
        """Dimension of the last successful embedding, or None before first probe."""
        with self._lock:
            return self._dimension

    # ── Availability ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model answers an embed call.

        Lazy and cached: the first call probes the daemon, subsequent
        calls return the cached boolean until ``reset_availability()``
        is invoked.
        """
        with self._lock:
            if self._available is not None:
                return self._available

        # Probe without holding the lock so other threads don't block
        # on the HTTP round-trip. Multiple concurrent callers may issue
        # duplicate probes — that's acceptable because probes are idempotent
        # and the worst case is one extra HTTP call during cold start.
        probe_ok, probe_dim = self._probe()

        with self._lock:
            # Last writer wins. If another thread already set this, we just
            # overwrite with our own result (which should match).
            self._available = probe_ok
            if probe_ok and probe_dim is not None:
                self._dimension = probe_dim
            return self._available

    def reset_availability(self) -> None:
        """Forget cached availability; the next call will probe again.

        Useful after a session boundary or when the caller explicitly
        wants to re-check whether Ollama came back online.
        """
        with self._lock:
            self._available = None

    # ── Embedding generation ───────────────────────────────────────────

    def embed(self, text: str) -> Optional[Tuple[bytes, int]]:
        """Generate an embedding for ``text`` as (blob, dim), or None if offline.

        Returns ``None`` (without raising) in any of these cases:

        - ``text`` is empty or whitespace-only
        - The daemon has already been marked unavailable this session
        - The HTTP request fails or times out
        - Ollama returns a non-200 status or a malformed body
        - The returned vector is empty

        Callers should treat ``None`` as "store NULL in the embedding
        column and let hybrid_search fall back to non-vector ranking".
        """
        if not text or not text.strip():
            return None

        if not self.is_available():
            return None

        try:
            vector = self._call_embed(text)
        except Exception as exc:  # pragma: no cover - covered via tests monkeypatching httpx
            logger.debug("Ollama embed call failed: %s", exc)
            with self._lock:
                self._available = False
            return None

        if not vector:
            # Daemon responded but gave us nothing useful; treat as offline.
            with self._lock:
                self._available = False
            return None

        try:
            blob = pack_embedding(vector)
        except ValueError as exc:
            logger.debug("Failed to pack embedding: %s", exc)
            return None

        dim = len(vector)
        with self._lock:
            self._dimension = dim

        return blob, dim

    # ── Private HTTP helpers ───────────────────────────────────────────

    # The probe string is intentionally short and neutral. A single token
    # is enough to verify both connectivity and that the model loads.
    _PROBE_TEXT = "probe"

    def _probe(self) -> Tuple[bool, Optional[int]]:
        """Run a single embed call to verify daemon + model health.

        Returns (ok, dim). ``ok=False`` on any exception or non-200;
        ``dim`` is the detected embedding dimension on success.
        """
        try:
            vector = self._call_embed(self._PROBE_TEXT)
        except Exception as exc:
            logger.debug("Ollama probe failed: %s", exc)
            return False, None

        if not vector:
            return False, None

        return True, len(vector)

    def _call_embed(self, text: str) -> Optional[list]:
        """POST ``text`` to Ollama's /api/embeddings endpoint.

        Isolated as a method so tests can monkeypatch ``httpx.Client``
        on the embeddings module without touching the surrounding
        state-management logic. Returns the raw embedding list, or
        ``None`` if the response is empty or malformed.

        Uses the ``/api/embeddings`` endpoint (single-prompt form) for
        broad compatibility with older Ollama builds. The newer
        ``/api/embed`` endpoint is equivalent but only available on
        Ollama 0.1.40+; sticking with the legacy path means Claudia
        works against any Ollama shipped in the last year.
        """
        import httpx  # noqa: PLC0415 - lazy import so unit tests can patch it

        url = f"{self._host}/api/embeddings"
        payload = {"model": self._model, "prompt": text}

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)

        if response.status_code != 200:
            logger.debug(
                "Ollama embeddings returned status %d: %s",
                response.status_code,
                response.text[:200],
            )
            return None

        try:
            data = response.json()
        except ValueError:
            logger.debug("Ollama embeddings returned non-JSON body")
            return None

        vector = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(vector, list) or not vector:
            return None

        return vector
