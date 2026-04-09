"""Integration tests for ClaudiaMemoryProvider.purge_old_soft_deletes
(Phase 2C.4) and threading of auto_merge_threshold into consolidate().
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import CommitmentDetector
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import LLMExtractor
from plugins.memory.claudia.retention import (
    DEFAULT_RETENTION_DAYS,
    RetentionResult,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


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


# ─── Purge tests ────────────────────────────────────────────────────────


class TestProviderPurge:
    def test_empty_db_returns_empty_result(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = p.purge_old_soft_deletes(timeout=5.0)
            assert isinstance(result, RetentionResult)
            assert result.total() == 0
        finally:
            p.shutdown()

    def test_old_soft_deleted_memory_purged(self, tmp_path):
        p = _provider(tmp_path)
        try:
            # Seed an old soft-deleted memory via the writer
            def _seed(conn):
                old = (
                    NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
                ).isoformat()
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at,
                        accessed_at, deleted_at
                    ) VALUES ('old', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'test', 'default',
                              ?, ?, ?)
                    """,
                    (NOW.isoformat(), NOW.isoformat(), old),
                )

            p._writer.enqueue_and_wait(_seed, timeout=5.0)

            result = p.purge_old_soft_deletes(timeout=5.0)
            assert result.memories_purged == 1

            # Verify the row is physically gone
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM memories"
                ).fetchone()
            assert row["n"] == 0
        finally:
            p.shutdown()

    def test_custom_retention_days_argument(self, tmp_path):
        """Explicit retention_days overrides config and default."""
        p = _provider(tmp_path)
        try:
            def _seed(conn):
                recent = (NOW - timedelta(days=5)).isoformat()
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at,
                        accessed_at, deleted_at
                    ) VALUES ('five days old', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'test', 'default',
                              ?, ?, ?)
                    """,
                    (NOW.isoformat(), NOW.isoformat(), recent),
                )

            p._writer.enqueue_and_wait(_seed, timeout=5.0)

            # With default retention (90d) this row survives
            result = p.purge_old_soft_deletes(timeout=5.0)
            assert result.memories_purged == 0

            # With explicit retention_days=3, the 5-day-old row is purged
            result = p.purge_old_soft_deletes(
                timeout=5.0, retention_days=3
            )
            assert result.memories_purged == 1
        finally:
            p.shutdown()

    def test_config_retention_days_honored(self, tmp_path):
        """retention_days from config is used when no explicit arg."""
        import json
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({"retention_days": 3})
        )

        p = _provider(tmp_path)
        try:
            def _seed(conn):
                five_days_ago = (NOW - timedelta(days=5)).isoformat()
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        source_type, source_ref, profile, created_at,
                        accessed_at, deleted_at
                    ) VALUES ('old', 'extracted', 0.7, 0.5, 0,
                              'conversation', 'test', 'default',
                              ?, ?, ?)
                    """,
                    (NOW.isoformat(), NOW.isoformat(), five_days_ago),
                )

            p._writer.enqueue_and_wait(_seed, timeout=5.0)

            # config retention_days=3, so 5-day-old row is purged
            result = p.purge_old_soft_deletes(timeout=5.0)
            assert result.memories_purged == 1
        finally:
            p.shutdown()

    def test_shutdown_provider_returns_empty(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        result = p.purge_old_soft_deletes(timeout=5.0)
        assert result.total() == 0


# ─── Config threshold flows into consolidate ────────────────────────────


class TestConsolidateConfigThreshold:
    def test_auto_merge_threshold_config_used(self, tmp_path):
        """Setting auto_merge_threshold to 0.99 makes consolidation
        MORE conservative — the Sarah Chen / Sarah C. alias-sharing
        pair still merges (score 0.95) because alias-match is above
        the fuzzy SequenceMatcher threshold, but a borderline
        typo-pair might not."""
        import json
        config_dir = tmp_path / "memory" / "claudia"
        config_dir.mkdir(parents=True)
        # Very high threshold: requires near-exact + alias
        (config_dir / "config.json").write_text(
            json.dumps({"auto_merge_threshold": 0.99})
        )

        p = _provider(tmp_path)
        try:
            from plugins.memory.claudia import entities

            def _seed(conn):
                # Two similar but not alias-sharing names:
                # "Jonathan Smithson" vs "Jonathan Smythe"
                # These would score ~0.92 via SequenceMatcher but
                # should NOT merge with threshold 0.99.
                entities.create_entity(conn, "person", "Jonathan Smithson")
                entities.create_entity(conn, "person", "Jonathan Smythson")

            p._writer.enqueue_and_wait(_seed, timeout=5.0)

            result = p.consolidate(timeout=5.0)
            # With 0.99 threshold, neither fuzzy match nor alias
            # overlap triggers → zero merges
            assert result.entities_merged == 0
        finally:
            p.shutdown()
