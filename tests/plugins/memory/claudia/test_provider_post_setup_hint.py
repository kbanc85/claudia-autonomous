"""Tests for provider.post_setup_hint (Phase 2D.3).

The hook that memory_setup.py calls after activating the plugin
to print next-steps guidance. Uses a bounded Ollama probe to
tailor the output.

This test inspects the text directly since it's a string meant
for humans — we check for the presence of expected guidance
keywords rather than exact string matching, so future wording
tweaks don't break tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import CommitmentDetector
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import LLMExtractor


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _NoOpExtractor(LLMExtractor):
    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _NoOpDetector(CommitmentDetector):
    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _TestProvider(ClaudiaMemoryProvider):
    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return _NoOpExtractor()

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return _NoOpDetector()


def _provider(tmp_path) -> _TestProvider:
    p = _TestProvider()
    p.initialize(
        session_id="test",
        claudia_home=str(tmp_path),
        platform="cli",
    )
    return p


# ─── Tests ──────────────────────────────────────────────────────────────


class TestPostSetupHint:
    def test_returns_string(self, tmp_path):
        p = _provider(tmp_path)
        try:
            hint = p.post_setup_hint()
            assert isinstance(hint, str)
            assert hint.strip() != ""
        finally:
            p.shutdown()

    def test_mentions_readme(self, tmp_path):
        """Every code path should point at the README for details."""
        p = _provider(tmp_path)
        try:
            hint = p.post_setup_hint()
            assert "README" in hint
        finally:
            p.shutdown()

    def test_mentions_installed_and_ready(self, tmp_path):
        p = _provider(tmp_path)
        try:
            hint = p.post_setup_hint()
            assert "installed" in hint.lower() or "ready" in hint.lower()
        finally:
            p.shutdown()


class TestPostSetupHintOllamaReachable:
    """Use subclass override of _probe_ollama_for_hint to inject
    deterministic Ollama states without making real HTTP calls."""

    def test_ollama_ok_with_both_models(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return (
                    "ok",
                    ["all-minilm:l6-v2", "qwen2.5:3b"],
                )

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            assert "Ollama reachable" in hint
            assert "demo" in hint
            # Should NOT tell the user to pull models they have
            assert "ollama pull all-minilm" not in hint
            assert "ollama pull qwen2.5" not in hint
        finally:
            p.shutdown()

    def test_ollama_ok_missing_embedding_model(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return ("ok", ["qwen2.5:3b"])  # missing embedding

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            assert "all-minilm:l6-v2" in hint
            assert "ollama pull" in hint
        finally:
            p.shutdown()

    def test_ollama_ok_missing_llm_model(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return ("ok", ["all-minilm:l6-v2"])  # missing LLM

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            assert "qwen2.5:3b" in hint
            assert "ollama pull" in hint
        finally:
            p.shutdown()

    def test_ollama_ok_no_models(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return ("ok", [])

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            assert "all-minilm:l6-v2" in hint
            assert "qwen2.5:3b" in hint
        finally:
            p.shutdown()


class TestPostSetupHintOllamaUnreachable:
    def test_unreachable_shows_install_link(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return ("unreachable", [])

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            assert "ollama.com" in hint
            assert "offline mode" in hint.lower()
            # Should mention that recall still works
            assert "FTS" in hint or "recall" in hint.lower()
        finally:
            p.shutdown()

    def test_unreachable_still_mentions_demo(self, tmp_path):
        class _ProbedProvider(_TestProvider):
            def _probe_ollama_for_hint(self):
                return ("unreachable", [])

        p = _ProbedProvider()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            hint = p.post_setup_hint()
            # Demo works offline so the hint should still mention it
            assert "demo" in hint
        finally:
            p.shutdown()


class TestProbeOllamaForHint:
    """The probe itself is fallible by design — all failures
    become the 'unreachable' case."""

    def test_never_raises(self, tmp_path):
        """Calling the probe against a non-running daemon shouldn't
        raise, even without any mocking."""
        p = _provider(tmp_path)
        try:
            # The real probe — should hit localhost:11434 or not.
            # Either way it returns a tuple, never raises.
            status, models = p._probe_ollama_for_hint()
            assert status in ("ok", "unreachable")
            assert isinstance(models, list)
        finally:
            p.shutdown()
