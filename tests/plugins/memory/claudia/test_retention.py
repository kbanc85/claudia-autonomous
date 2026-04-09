"""Unit tests for plugins/memory/claudia/retention.py (Phase 2C.4).

Retention purge permanently removes rows whose ``deleted_at``
timestamp is older than a configurable retention window. Runs
across all four soft-delete-bearing tables: memories, entities,
relationships, commitments.

Covered:

- ``RetentionResult`` dataclass defaults and fields
- ``purge_old_soft_deletes``:
    * Empty DB returns zeros
    * Old soft-deleted memory removed
    * Recent soft-deleted memory preserved (within retention)
    * Non-deleted (deleted_at IS NULL) rows never touched
    * Entities purged
    * Relationships purged
    * Commitments purged
    * Per-table counts in result
    * Profile isolation
    * retention_days=0 removes everything soft-deleted
    * negative retention_days is clamped to 0 (defensive)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.claudia import commitments, entities, schema
from plugins.memory.claudia.retention import (
    DEFAULT_RETENTION_DAYS,
    RetentionResult,
    purge_old_soft_deletes,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = schema.ensure_database(tmp_path / "retention.db")
    yield db
    db.close()


def _seed_memory(conn, content, *, deleted_at=None, profile="default"):
    now_iso = NOW.isoformat()
    cur = conn.execute(
        """
        INSERT INTO memories (
            content, origin, confidence, importance, access_count,
            source_type, source_ref, profile, created_at, accessed_at,
            deleted_at
        ) VALUES (?, 'extracted', 0.7, 0.5, 0, 'conversation', 'test',
                  ?, ?, ?, ?)
        """,
        (
            content,
            profile,
            now_iso,
            now_iso,
            deleted_at.isoformat() if deleted_at else None,
        ),
    )
    return cur.lastrowid


def _count_all_rows(conn, table):
    """Count ALL rows including soft-deleted ones (raw count)."""
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return row["n"]


# ─── RetentionResult ────────────────────────────────────────────────────


class TestRetentionResult:
    def test_defaults(self):
        r = RetentionResult()
        assert r.memories_purged == 0
        assert r.entities_purged == 0
        assert r.relationships_purged == 0
        assert r.commitments_purged == 0

    def test_fields_settable(self):
        r = RetentionResult(
            memories_purged=5,
            entities_purged=2,
            relationships_purged=3,
            commitments_purged=1,
        )
        assert r.memories_purged == 5

    def test_total_helper(self):
        """Optional helper: total() returns the sum across tables."""
        r = RetentionResult(
            memories_purged=5, entities_purged=2,
            relationships_purged=3, commitments_purged=1,
        )
        assert r.total() == 11


class TestDefaultRetentionDays:
    def test_positive(self):
        assert DEFAULT_RETENTION_DAYS > 0

    def test_reasonable_default(self):
        """Retention should be long enough for meaningful debugging
        (at least 30 days) but not so long it defeats the purpose
        (under a year)."""
        assert 30 <= DEFAULT_RETENTION_DAYS <= 365


# ─── purge_old_soft_deletes ────────────────────────────────────────────


class TestPurgeMemories:
    def test_empty_db_returns_zeros(self, conn):
        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 0

    def test_old_soft_deleted_removed(self, conn):
        old_deleted = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        mid = _seed_memory(conn, "old", deleted_at=old_deleted)
        assert _count_all_rows(conn, "memories") == 1

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 1
        assert _count_all_rows(conn, "memories") == 0

    def test_recent_soft_deleted_preserved(self, conn):
        """Soft-deleted within retention window stays."""
        recent = NOW - timedelta(days=1)
        mid = _seed_memory(conn, "recent", deleted_at=recent)

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 0
        assert _count_all_rows(conn, "memories") == 1

    def test_non_deleted_never_touched(self, conn):
        """deleted_at IS NULL rows must never be purged."""
        mid = _seed_memory(conn, "live", deleted_at=None)
        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 0
        assert _count_all_rows(conn, "memories") == 1

    def test_mix_of_old_recent_live(self, conn):
        """Only the old-soft-deleted row is purged."""
        old = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 5)
        _seed_memory(conn, "old_deleted", deleted_at=old)
        _seed_memory(conn, "recent_deleted", deleted_at=NOW - timedelta(days=2))
        _seed_memory(conn, "live", deleted_at=None)

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 1
        # 2 rows survive
        assert _count_all_rows(conn, "memories") == 2


class TestPurgeEntities:
    def test_old_soft_deleted_entity_removed(self, conn):
        ent = entities.create_entity(conn, "person", "Sarah", now=NOW)
        old_deleted = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        entities.soft_delete_entity(conn, ent.id, now=old_deleted)

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.entities_purged == 1
        assert _count_all_rows(conn, "entities") == 0

    def test_recent_soft_deleted_entity_preserved(self, conn):
        ent = entities.create_entity(conn, "person", "Sarah", now=NOW)
        entities.soft_delete_entity(
            conn, ent.id, now=NOW - timedelta(days=1)
        )
        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.entities_purged == 0


class TestPurgeRelationships:
    def test_old_soft_deleted_relationship_removed(self, conn):
        a = entities.create_entity(conn, "person", "A", now=NOW)
        b = entities.create_entity(conn, "person", "B", now=NOW)
        rel = entities.create_relationship(
            conn, a.id, b.id, "knows", now=NOW
        )
        old_deleted = (
            NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        ).isoformat()
        conn.execute(
            "UPDATE relationships SET deleted_at = ? WHERE id = ?",
            (old_deleted, rel.id),
        )

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.relationships_purged == 1


class TestPurgeCommitments:
    def test_old_soft_deleted_commitment_removed(self, conn):
        c = commitments.create_commitment(conn, "do x", now=NOW)
        old_deleted = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        commitments.soft_delete_commitment(conn, c.id, now=old_deleted)

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.commitments_purged == 1

    def test_recent_soft_deleted_commitment_preserved(self, conn):
        c = commitments.create_commitment(conn, "do x", now=NOW)
        commitments.soft_delete_commitment(
            conn, c.id, now=NOW - timedelta(days=5)
        )
        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.commitments_purged == 0


class TestPurgeProfileIsolation:
    def test_other_profile_untouched(self, conn):
        old = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        _seed_memory(conn, "user_a mem", deleted_at=old, profile="user_a")
        _seed_memory(conn, "user_b mem", deleted_at=old, profile="user_b")

        # Only purge user_a
        result = purge_old_soft_deletes(conn, profile="user_a", now=NOW)
        assert result.memories_purged == 1
        # user_b row still there
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE profile = 'user_b'"
        ).fetchone()
        assert row["n"] == 1


class TestPurgeRetentionDaysParam:
    def test_custom_retention_days(self, conn):
        """retention_days=5 purges rows soft-deleted 6+ days ago."""
        _seed_memory(
            conn, "6 days", deleted_at=NOW - timedelta(days=6)
        )
        _seed_memory(
            conn, "4 days", deleted_at=NOW - timedelta(days=4)
        )
        result = purge_old_soft_deletes(
            conn, profile="default", now=NOW, retention_days=5
        )
        assert result.memories_purged == 1
        # Recent row survives
        assert _count_all_rows(conn, "memories") == 1

    def test_retention_days_zero_purges_all_soft_deleted(self, conn):
        """retention_days=0 purges everything soft-deleted."""
        _seed_memory(conn, "just now", deleted_at=NOW - timedelta(seconds=1))
        result = purge_old_soft_deletes(
            conn, profile="default", now=NOW, retention_days=0
        )
        assert result.memories_purged == 1

    def test_negative_retention_days_clamped(self, conn):
        """Defensive: negative retention clamps to 0 (same as "purge
        everything soft-deleted")."""
        _seed_memory(conn, "x", deleted_at=NOW - timedelta(seconds=1))
        result = purge_old_soft_deletes(
            conn, profile="default", now=NOW, retention_days=-10
        )
        assert result.memories_purged == 1


class TestPurgeTotal:
    def test_across_all_tables(self, conn):
        """A single purge call hits all four tables."""
        old = NOW - timedelta(days=DEFAULT_RETENTION_DAYS + 10)

        # Memory
        _seed_memory(conn, "old", deleted_at=old)
        # Entity
        ent = entities.create_entity(conn, "person", "A", now=NOW)
        entities.soft_delete_entity(conn, ent.id, now=old)
        # Relationship — need another live entity
        b = entities.create_entity(conn, "person", "B", now=NOW)
        rel = entities.create_relationship(
            conn, b.id, b.id,  # self-ref just to make a row
            "reminds_of", now=NOW,
        )
        conn.execute(
            "UPDATE relationships SET deleted_at = ? WHERE id = ?",
            (old.isoformat(), rel.id),
        )
        # Commitment
        c = commitments.create_commitment(conn, "do x", now=NOW)
        commitments.soft_delete_commitment(conn, c.id, now=old)

        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.memories_purged == 1
        assert result.entities_purged == 1
        assert result.relationships_purged == 1
        assert result.commitments_purged == 1
        assert result.total() == 4
