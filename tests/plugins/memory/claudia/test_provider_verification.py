"""Integration tests for ClaudiaMemoryProvider.verify (Phase 2B.4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import CommitmentDetector
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import LLMExtractor
from plugins.memory.claudia.verification import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_STALE_DAYS,
    VerificationResult,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── Test doubles ───────────────────────────────────────────────────────


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


def _insert_old_memory(
    p,
    content,
    *,
    origin="extracted",
    verification_state="pending",
    days_old=DEFAULT_HALF_LIFE_DAYS * 5,
    confidence=0.7,
):
    """Insert a memory via the writer queue with an old accessed_at."""
    accessed = (NOW - timedelta(days=days_old)).isoformat()
    profile = p._profile

    def _job(conn):
        cur = conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                verification, source_type, source_ref, profile,
                created_at, accessed_at
            ) VALUES (?, ?, ?, 0.5, 0, ?, 'conversation', 'test', ?, ?, ?)
            """,
            (
                content,
                origin,
                confidence,
                verification_state,
                profile,
                NOW.isoformat(),
                accessed,
            ),
        )
        return int(cur.lastrowid)

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


def _read_memory(p, memory_id):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        return conn.execute(
            """
            SELECT id, confidence, verification, origin, accessed_at
            FROM memories WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()


# ─── Tests ──────────────────────────────────────────────────────────────


class TestVerifyBasic:
    def test_empty_db_returns_zeros(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = p.verify(timeout=5.0)
            assert isinstance(result, VerificationResult)
            assert result.decayed_count == 0
            assert result.flagged_stale_count == 0
        finally:
            p.shutdown()

    def test_decays_old_extracted_memory(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _insert_old_memory(
                p, "old fact", origin="extracted",
                days_old=DEFAULT_HALF_LIFE_DAYS * 5,
            )
            result = p.verify(timeout=5.0)
            assert result.decayed_count == 1
            row = _read_memory(p, mid)
            assert row["confidence"] < 0.7
        finally:
            p.shutdown()

    def test_preserves_user_stated(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _insert_old_memory(
                p, "user fact", origin="user_stated",
                confidence=1.0,
                days_old=DEFAULT_HALF_LIFE_DAYS * 10,
            )
            p.verify(timeout=5.0)
            row = _read_memory(p, mid)
            assert row["confidence"] == 1.0
        finally:
            p.shutdown()

    def test_flags_stale_pending(self, tmp_path):
        p = _provider(tmp_path)
        try:
            mid = _insert_old_memory(
                p, "stale fact",
                verification_state="pending",
                days_old=DEFAULT_STALE_DAYS + 10,
            )
            result = p.verify(timeout=5.0)
            assert result.flagged_stale_count == 1
            row = _read_memory(p, mid)
            assert row["verification"] == "flagged"
        finally:
            p.shutdown()

    def test_idempotent(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _insert_old_memory(
                p, "old", origin="extracted",
                days_old=DEFAULT_HALF_LIFE_DAYS * 5,
            )
            _insert_old_memory(
                p, "stale", verification_state="pending",
                days_old=DEFAULT_STALE_DAYS + 10,
            )
            first = p.verify(timeout=5.0)
            second = p.verify(timeout=5.0)
            assert second.decayed_count == 0
            assert second.flagged_stale_count == 0
        finally:
            p.shutdown()

    def test_shutdown_returns_empty(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        result = p.verify(timeout=5.0)
        assert result.decayed_count == 0

    def test_profile_isolated(self, tmp_path):
        p = _provider(tmp_path, user_id="user_a")
        try:
            assert p._profile == "user_a"
            _insert_old_memory(
                p, "user_a memory", origin="extracted",
                days_old=DEFAULT_HALF_LIFE_DAYS * 5,
            )

            # Insert into user_b via raw writer job
            def _seed_other(conn):
                conn.execute(
                    """
                    INSERT INTO memories (
                        content, origin, confidence, importance, access_count,
                        verification, source_type, source_ref, profile,
                        created_at, accessed_at
                    ) VALUES ('user_b memory', 'extracted', 0.7, 0.5, 0,
                              'pending', 'conversation', 'test', 'user_b', ?, ?)
                    """,
                    (
                        NOW.isoformat(),
                        (NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 10)).isoformat(),
                    ),
                )

            p._writer.enqueue_and_wait(_seed_other, timeout=5.0)

            result = p.verify(timeout=5.0)
            assert result.decayed_count == 1  # only user_a's

            # user_b's memory is untouched
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT confidence FROM memories WHERE profile = 'user_b'"
                ).fetchone()
            assert row["confidence"] == 0.7
        finally:
            p.shutdown()
