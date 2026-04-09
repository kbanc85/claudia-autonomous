"""Unit tests for plugins/memory/claudia/embeddings.py (Phase 2A.2b).

Covers:

- BLOB encoding helpers (pack_embedding / unpack_embedding) round-trip
  and error handling for corrupt input.
- Host resolution precedence: explicit arg > env var > default.
- Lazy availability: no network I/O at construction time.
- Probe caching: success caches True, failure caches False, both
  survive subsequent calls without re-probing.
- ``reset_availability`` forces a re-probe.
- ``embed()`` happy path returns (blob, dim) matching pack layout.
- ``embed()`` offline paths: empty text, downed daemon, non-200,
  malformed JSON, missing "embedding" key, empty vector.
- Stickiness: once ``embed()`` marks the embedder unavailable
  mid-session, further calls return None without re-probing.

HTTP is isolated by subclassing ``OllamaEmbedder`` and overriding
``_call_embed``. That gives us deterministic, fast tests with no
network dependency and no coupling to httpx's internals.
"""

from __future__ import annotations

import struct

import pytest

from plugins.memory.claudia.embeddings import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    ENV_HOST,
    OllamaEmbedder,
    pack_embedding,
    unpack_embedding,
)


# ─── BLOB encoding helpers ───────────────────────────────────────────────


class TestPackEmbedding:
    def test_roundtrip_preserves_values(self):
        original = [0.1, -0.2, 3.14159, 42.0, -0.0, 1e-5]
        blob = pack_embedding(original)
        restored = unpack_embedding(blob)
        assert len(restored) == len(original)
        for got, want in zip(restored, original):
            assert got == pytest.approx(want, rel=1e-5, abs=1e-6)

    def test_blob_length_is_dim_times_four(self):
        vec = [0.0] * 384
        assert len(pack_embedding(vec)) == 384 * 4

    def test_accepts_tuple_input(self):
        blob = pack_embedding((0.5, -0.5))
        assert unpack_embedding(blob) == pytest.approx([0.5, -0.5])

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            pack_embedding(None)

    def test_rejects_empty_vector(self):
        with pytest.raises(ValueError):
            pack_embedding([])

    def test_uses_little_endian_float32(self):
        # Spot-check the byte layout matches <f format explicitly.
        blob = pack_embedding([1.0])
        assert blob == struct.pack("<f", 1.0)


class TestUnpackEmbedding:
    def test_empty_blob_returns_empty_list(self):
        assert unpack_embedding(b"") == []

    def test_none_blob_returns_empty_list(self):
        assert unpack_embedding(None) == []  # type: ignore[arg-type]

    def test_corrupt_length_raises(self):
        # 5 bytes is not a multiple of 4, so cannot be a float32 array.
        with pytest.raises(ValueError):
            unpack_embedding(b"\x00\x00\x00\x00\x01")


# ─── Host resolution ─────────────────────────────────────────────────────


class TestHostResolution:
    def test_default_host_when_no_override(self, monkeypatch):
        monkeypatch.delenv(ENV_HOST, raising=False)
        embedder = OllamaEmbedder()
        assert embedder.host == DEFAULT_HOST

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv(ENV_HOST, "http://remote-ollama:9999")
        embedder = OllamaEmbedder()
        assert embedder.host == "http://remote-ollama:9999"

    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(ENV_HOST, "http://env-host:1111")
        embedder = OllamaEmbedder(host="http://explicit:2222")
        assert embedder.host == "http://explicit:2222"

    def test_host_trailing_slash_stripped(self):
        embedder = OllamaEmbedder(host="http://ollama:11434/")
        assert embedder.host == "http://ollama:11434"

    def test_default_model(self):
        assert OllamaEmbedder().model == DEFAULT_MODEL


# ─── Fake embedder for controllable HTTP behaviour ───────────────────────


class _FakeEmbedder(OllamaEmbedder):
    """OllamaEmbedder with ``_call_embed`` swapped out for a scripted fake.

    Each instance carries a script — a list of values returned by
    successive ``_call_embed`` calls. Entries that are exceptions are
    raised; other entries are returned directly. The fake also records
    how many times ``_call_embed`` ran so tests can verify caching.
    """

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self._script = list(script)
        self.call_count = 0

    def _call_embed(self, text):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            raise AssertionError(
                f"_call_embed invoked more times than script length; text={text!r}"
            )
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# ─── Lazy availability ──────────────────────────────────────────────────


class TestLazyAvailability:
    def test_construction_is_offline(self, monkeypatch):
        """No network or _call_embed invocation should happen at construction."""

        def boom(self, text):
            raise AssertionError("_call_embed must not be called during construction")

        monkeypatch.setattr(OllamaEmbedder, "_call_embed", boom)
        embedder = OllamaEmbedder()  # should not raise
        assert embedder.dimension is None

    def test_is_available_probes_once_on_success(self):
        probe_vector = [0.1, 0.2, 0.3]
        embedder = _FakeEmbedder(script=[probe_vector])
        assert embedder.is_available() is True
        assert embedder.call_count == 1
        # Second call must reuse the cached result.
        assert embedder.is_available() is True
        assert embedder.call_count == 1
        assert embedder.dimension == 3

    def test_is_available_probes_once_on_failure(self):
        embedder = _FakeEmbedder(script=[ConnectionError("no daemon")])
        assert embedder.is_available() is False
        assert embedder.call_count == 1
        # Still cached as False — don't hammer a downed daemon.
        assert embedder.is_available() is False
        assert embedder.call_count == 1
        assert embedder.dimension is None

    def test_is_available_false_when_probe_returns_empty(self):
        embedder = _FakeEmbedder(script=[None])
        assert embedder.is_available() is False

    def test_reset_availability_forces_reprobe(self):
        embedder = _FakeEmbedder(
            script=[ConnectionError("down"), [0.1, 0.2]]
        )
        assert embedder.is_available() is False
        embedder.reset_availability()
        assert embedder.is_available() is True
        assert embedder.call_count == 2
        assert embedder.dimension == 2


# ─── embed() behaviour ───────────────────────────────────────────────────


class TestEmbed:
    def test_happy_path_returns_blob_and_dim(self):
        vector = [0.25, -0.5, 0.75]
        # First call is the probe, second is the real embed — both use the
        # same vector to keep the test simple. In practice the probe text
        # differs from user text but the daemon returns one vector per call.
        embedder = _FakeEmbedder(script=[vector, vector])
        result = embedder.embed("hello world")

        assert result is not None
        blob, dim = result
        assert dim == 3
        assert unpack_embedding(blob) == pytest.approx(vector, rel=1e-5)
        assert embedder.dimension == 3

    def test_empty_text_returns_none_without_probing(self):
        embedder = _FakeEmbedder(script=[])  # script empty — any call fails the assert
        assert embedder.embed("") is None
        assert embedder.embed("   \n\t") is None
        assert embedder.call_count == 0

    def test_returns_none_when_daemon_unavailable(self):
        embedder = _FakeEmbedder(script=[ConnectionError("no daemon")])
        assert embedder.embed("hello") is None
        assert embedder.call_count == 1

    def test_none_result_is_sticky_for_session(self):
        """Once availability is False, embed() short-circuits without HTTP."""
        embedder = _FakeEmbedder(script=[ConnectionError("down")])
        assert embedder.embed("first") is None
        # Second embed must NOT invoke _call_embed again.
        assert embedder.embed("second") is None
        assert embedder.call_count == 1

    def test_mid_session_failure_marks_unavailable(self):
        """Probe succeeds, then embed raises — subsequent embed should short-circuit."""
        probe_vector = [0.1, 0.2]
        embedder = _FakeEmbedder(
            script=[probe_vector, ConnectionError("daemon died mid-session")]
        )
        # Warm up with a successful probe via is_available.
        assert embedder.is_available() is True
        # The actual embed call fails — should be marked unavailable.
        assert embedder.embed("payload") is None
        # Third call must not retry.
        assert embedder.embed("another") is None
        assert embedder.call_count == 2  # probe + failed embed, nothing after

    def test_empty_vector_response_marks_unavailable(self):
        """A 200 with empty vector is treated as a broken daemon."""
        probe_vector = [0.1, 0.2]
        embedder = _FakeEmbedder(script=[probe_vector, None])
        assert embedder.is_available() is True
        # _call_embed returns None — treat as offline.
        assert embedder.embed("payload") is None
        assert embedder.embed("second") is None
        assert embedder.call_count == 2


# ─── Integration: pack/unpack symmetry with real dimensions ─────────────


class TestDimensionPreservation:
    @pytest.mark.parametrize("dim", [3, 128, 384, 768, 1536])
    def test_roundtrip_various_dims(self, dim):
        vec = [i * 0.001 for i in range(dim)]
        blob = pack_embedding(vec)
        restored = unpack_embedding(blob)
        assert len(restored) == dim
        # pytest.approx on a list does element-wise comparison.
        assert restored == pytest.approx(vec, rel=1e-4, abs=1e-5)

    def test_embedder_tracks_dimension_after_embed(self):
        vector = [0.0] * 384
        embedder = _FakeEmbedder(script=[vector, vector])
        embedder.embed("warm up the probe")
        assert embedder.dimension == 384
