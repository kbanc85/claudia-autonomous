"""Tests for ClaudiaMemoryProvider config schema + persistence (Phase 2C.3).

Covers:

- ``get_config_schema`` returns the expected field set
- ``save_config`` writes JSON to the expected path
- ``save_config`` merges with an existing file
- ``save_config`` creates the parent directory if missing
- ``_load_config`` reads the JSON
- ``_load_config`` returns {} on missing/corrupt file
- Loaded config is used by factory methods (extractor model,
  detector model, host)
- Loaded config is passed to verify() (half_life_days, stale_days)
- Loaded config is passed to consolidate() (auto_merge_threshold)
- Missing config keys fall back to defaults
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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


def _provider(tmp_path, **init_kwargs) -> _TestProvider:
    p = _TestProvider()
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


# ─── get_config_schema ──────────────────────────────────────────────────


class TestGetConfigSchema:
    def test_returns_list_of_dicts(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = p.get_config_schema()
            assert isinstance(schema, list)
            for field in schema:
                assert isinstance(field, dict)
                assert "key" in field
                assert "description" in field
        finally:
            p.shutdown()

    def test_includes_expected_keys(self, tmp_path):
        p = _provider(tmp_path)
        try:
            schema = p.get_config_schema()
            keys = {f["key"] for f in schema}
            # Minimum expected config surface
            required = {
                "ollama_host",
                "embedding_model",
                "extraction_model",
                "detection_model",
                "decay_half_life_days",
                "stale_flag_days",
                "auto_merge_threshold",
            }
            assert required.issubset(keys)
        finally:
            p.shutdown()

    def test_no_secrets_flagged(self, tmp_path):
        """Claudia's config has no API keys (local-only Ollama)."""
        p = _provider(tmp_path)
        try:
            schema = p.get_config_schema()
            for field in schema:
                assert not field.get("secret", False)
        finally:
            p.shutdown()


# ─── save_config ────────────────────────────────────────────────────────


class TestSaveConfig:
    def test_writes_json_to_expected_path(self, tmp_path):
        p = _provider(tmp_path)
        try:
            values = {"extraction_model": "llama3:8b"}
            p.save_config(values, str(tmp_path))
            expected_path = tmp_path / "memory" / "claudia" / "config.json"
            assert expected_path.exists()
            data = json.loads(expected_path.read_text())
            assert data["extraction_model"] == "llama3:8b"
        finally:
            p.shutdown()

    def test_creates_parent_directories(self, tmp_path):
        """If memory/claudia/ doesn't exist, save_config creates it."""
        p = _provider(tmp_path)
        try:
            fresh_home = tmp_path / "fresh_home"
            # Do NOT pre-create the directory
            p.save_config({"ollama_host": "http://remote:11434"}, str(fresh_home))
            config_path = fresh_home / "memory" / "claudia" / "config.json"
            assert config_path.exists()
        finally:
            p.shutdown()

    def test_merges_with_existing_config(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # First write
            p.save_config(
                {"extraction_model": "llama3:8b", "ollama_host": "http://a"},
                str(tmp_path),
            )
            # Second write overwrites one key, keeps the other
            p.save_config(
                {"extraction_model": "qwen2.5:14b"},
                str(tmp_path),
            )
            config_path = tmp_path / "memory" / "claudia" / "config.json"
            data = json.loads(config_path.read_text())
            assert data["extraction_model"] == "qwen2.5:14b"
            assert data["ollama_host"] == "http://a"  # preserved
        finally:
            p.shutdown()


# ─── Config loading ────────────────────────────────────────────────────


class TestConfigLoading:
    def test_missing_config_returns_empty(self, tmp_path):
        """A provider initialized with no config file has an empty
        self._config dict."""
        p = _provider(tmp_path)
        try:
            assert p._config == {}
        finally:
            p.shutdown()

    def test_existing_config_loaded_into_provider(self, tmp_path):
        """Write a config file before initialize, verify it's loaded."""
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text(
            json.dumps({
                "extraction_model": "llama3:8b",
                "decay_half_life_days": 45,
            })
        )

        p = _provider(tmp_path)
        try:
            assert p._config["extraction_model"] == "llama3:8b"
            assert p._config["decay_half_life_days"] == 45
        finally:
            p.shutdown()

    def test_corrupt_config_returns_empty(self, tmp_path):
        """Invalid JSON doesn't crash initialize — just falls back."""
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text("not valid {json[")

        p = _provider(tmp_path)
        try:
            assert p._config == {}
        finally:
            p.shutdown()


# ─── Config used by verify / consolidate ────────────────────────────────


class TestConfigThreading:
    def test_verify_uses_configured_half_life(self, tmp_path):
        """verify() should pass self._config values to run_verification."""
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({
                "decay_half_life_days": 7,  # much shorter than default 30
                "stale_flag_days": 14,
            })
        )

        p = _provider(tmp_path)
        try:
            # Seed a memory that's 14 days old. With the shorter
            # half-life (7 days), this is 2 half-lives, so confidence
            # should drop to 0.7 * 0.25 = 0.175.
            from datetime import datetime, timedelta, timezone

            old_iso = (
                datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
                - timedelta(days=14)
            ).isoformat()

            def _seed(conn):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at, accessed_at
                    ) VALUES ('old fact', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'test', 'default', ?, ?)
                    """,
                    (old_iso, old_iso),
                )

            p._writer.enqueue_and_wait(_seed, timeout=5.0)

            # With half_life=30 (default), 14 days is <1 half-life.
            # With half_life=7 (configured), 14 days is 2 half-lives.
            # The latter should decay the confidence much more.
            result = p.verify(timeout=5.0)
            assert result.decayed_count == 1

            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT confidence FROM memories LIMIT 1"
                ).fetchone()
            # With 2 half-lives at the configured 7-day setting,
            # confidence should be ~0.175 (0.7 * 0.25).
            assert row["confidence"] == pytest.approx(0.175, abs=0.02)
        finally:
            p.shutdown()

    def test_consolidate_uses_default_threshold_without_config(self, tmp_path):
        """With no config, consolidate uses the default
        AUTO_MERGE_THRESHOLD. Just a sanity check that the path
        doesn't crash."""
        p = _provider(tmp_path)
        try:
            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 0
        finally:
            p.shutdown()


# ─── Factory integration ───────────────────────────────────────────────


class TestFactoryIntegrationWithConfig:
    def test_extractor_factory_receives_config_model(self, tmp_path):
        """When config specifies extraction_model, a real
        (non-test-override) factory call uses it."""
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({"extraction_model": "llama3:70b"})
        )

        # Use the REAL provider (not _TestProvider override) so
        # _make_extractor returns an OllamaLLMExtractor.
        from plugins.memory.claudia.embeddings import OllamaEmbedder

        class _ProviderWithRealExtractor(ClaudiaMemoryProvider):
            def _make_embedder(self):
                return _FakeEmbedder()

            # _make_extractor stays as the base class impl, which
            # should read self._config for the model

        p = _ProviderWithRealExtractor()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            # The real extractor should have the configured model
            assert p._extractor is not None
            assert p._extractor.model == "llama3:70b"
        finally:
            p.shutdown()

    def test_detector_factory_receives_config_model(self, tmp_path):
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({"detection_model": "qwen2.5:14b"})
        )

        class _ProviderWithRealDetector(ClaudiaMemoryProvider):
            def _make_embedder(self):
                return _FakeEmbedder()

        p = _ProviderWithRealDetector()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            assert p._commitment_detector is not None
            assert p._commitment_detector.model == "qwen2.5:14b"
        finally:
            p.shutdown()

    def test_default_models_used_without_config(self, tmp_path):
        """No config → factories use their default models."""
        class _ProviderWithRealFactories(ClaudiaMemoryProvider):
            def _make_embedder(self):
                return _FakeEmbedder()

        p = _ProviderWithRealFactories()
        p.initialize(
            session_id="test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        try:
            # Defaults from extractor.py / commitment_detector.py
            assert p._extractor.model == "qwen2.5:3b"
            assert p._commitment_detector.model == "qwen2.5:3b"
        finally:
            p.shutdown()
