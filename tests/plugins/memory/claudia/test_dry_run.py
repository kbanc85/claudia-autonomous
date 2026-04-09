"""Tests for dry_run behavior on consolidate and purge (Phase 2D.6).

dry_run=True runs the full scan/counting logic but does NOT
commit any destructive changes. The returned counts must match
what a real run on the same state would produce.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.claudia import (
    commitments,
    entities,
    schema,
)
from plugins.memory.claudia.consolidation import (
    ConsolidationResult,
    run_consolidation,
)
from plugins.memory.claudia.retention import (
    DEFAULT_RETENTION_DAYS,
    RetentionResult,
    purge_old_soft_deletes,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = schema.ensure_database(tmp_path / "dry_run.db")
    yield db
    db.close()


def _count_live_entities(conn, *, profile="default") -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM entities "
        "WHERE profile = ? AND deleted_at IS NULL",
        (profile,),
    ).fetchone()
    return row["n"]


def _count_all_memories(conn) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
    return row["n"]


# ─── Consolidation dry-run ─────────────────────────────────────────────


class TestConsolidationDryRun:
    def test_dry_run_leaves_db_unchanged(self, conn):
        a = entities.create_entity(
            conn, "person", "Sarah Chen",
            aliases=["schen"], importance=0.9, now=NOW,
        )
        b = entities.create_entity(
            conn, "person", "Sarah C.",
            aliases=["schen"], importance=0.3, now=NOW,
        )

        # Real run would merge these two
        result = run_consolidation(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert result.dry_run is True
        assert result.entities_merged == 1

        # But NO entities were soft-deleted
        assert _count_live_entities(conn) == 2
        # Both specific ids still alive
        assert entities.get_entity(conn, a.id) is not None
        assert entities.get_entity(conn, b.id) is not None

    def test_dry_run_count_matches_real_run(self, conn):
        entities.create_entity(
            conn, "person", "Sarah Chen",
            aliases=["schen"], importance=0.9, now=NOW,
        )
        entities.create_entity(
            conn, "person", "Sarah C.",
            aliases=["schen"], importance=0.3, now=NOW,
        )
        entities.create_entity(
            conn, "person", "Bob Smith",
            aliases=["bsmith"], importance=0.5, now=NOW,
        )
        entities.create_entity(
            conn, "person", "Bob S.",
            aliases=["bsmith"], importance=0.4, now=NOW,
        )

        # First: dry-run
        dry = run_consolidation(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert dry.entities_merged == 2

        # Verify: DB still has 4 live entities
        assert _count_live_entities(conn) == 4

        # Then: real run on the same state
        real = run_consolidation(
            conn, profile="default", now=NOW, dry_run=False
        )
        assert real.entities_merged == dry.entities_merged
        assert real.dry_run is False

        # Now the DB reflects the merge
        assert _count_live_entities(conn) == 2

    def test_dry_run_counts_linkable_commitments(self, conn):
        entities.create_entity(conn, "person", "Sarah", now=NOW)
        commitments.create_commitment(
            conn, "send the proposal to Sarah", now=NOW
        )
        commitments.create_commitment(
            conn, "email Bob about Q4", now=NOW
        )

        dry = run_consolidation(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert dry.commitments_linked == 1

        # The commitment's target_entity_id is still NULL
        row = conn.execute(
            "SELECT target_entity_id FROM commitments "
            "WHERE content = 'send the proposal to Sarah'"
        ).fetchone()
        assert row["target_entity_id"] is None

    def test_dry_run_default_is_false(self, conn):
        """Backward compat: omitting dry_run defaults to a real run."""
        entities.create_entity(
            conn, "person", "Sarah Chen",
            aliases=["schen"], now=NOW,
        )
        entities.create_entity(
            conn, "person", "Sarah C.",
            aliases=["schen"], now=NOW,
        )

        result = run_consolidation(conn, profile="default", now=NOW)
        assert result.dry_run is False
        assert result.entities_merged == 1
        # Actually merged
        assert _count_live_entities(conn) == 1

    def test_empty_db_dry_run_returns_zeros(self, conn):
        result = run_consolidation(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert result.entities_merged == 0
        assert result.commitments_linked == 0
        assert result.dry_run is True


# ─── Retention dry-run ─────────────────────────────────────────────────


def _seed_old_soft_deleted_memory(conn, content, *, days_ago):
    old = (NOW - timedelta(days=days_ago)).isoformat()
    conn.execute(
        """
        INSERT INTO memories (
            content, origin, confidence, importance, access_count,
            source_type, source_ref, profile, created_at, accessed_at,
            deleted_at
        ) VALUES (?, 'extracted', 0.7, 0.5, 0, 'conversation', 'test',
                  'default', ?, ?, ?)
        """,
        (content, NOW.isoformat(), NOW.isoformat(), old),
    )


class TestRetentionDryRun:
    def test_dry_run_leaves_rows_in_place(self, conn):
        _seed_old_soft_deleted_memory(
            conn, "old", days_ago=DEFAULT_RETENTION_DAYS + 10
        )
        assert _count_all_memories(conn) == 1

        result = purge_old_soft_deletes(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert result.dry_run is True
        assert result.memories_purged == 1

        # Row is still physically there
        assert _count_all_memories(conn) == 1

    def test_dry_run_count_matches_real_run(self, conn):
        for i in range(5):
            _seed_old_soft_deleted_memory(
                conn, f"old {i}",
                days_ago=DEFAULT_RETENTION_DAYS + 10,
            )

        dry = purge_old_soft_deletes(
            conn, profile="default", now=NOW, dry_run=True
        )
        assert dry.memories_purged == 5

        # DB unchanged
        assert _count_all_memories(conn) == 5

        real = purge_old_soft_deletes(
            conn, profile="default", now=NOW, dry_run=False
        )
        assert real.memories_purged == dry.memories_purged

        # Now gone
        assert _count_all_memories(conn) == 0

    def test_dry_run_respects_retention_days(self, conn):
        _seed_old_soft_deleted_memory(conn, "10 days", days_ago=10)
        _seed_old_soft_deleted_memory(conn, "30 days", days_ago=30)
        _seed_old_soft_deleted_memory(conn, "100 days", days_ago=100)

        # With retention_days=20, only the 30-day and 100-day
        # rows count as purgeable
        result = purge_old_soft_deletes(
            conn, profile="default", now=NOW,
            retention_days=20, dry_run=True,
        )
        assert result.memories_purged == 2
        assert _count_all_memories(conn) == 3

    def test_default_dry_run_false(self, conn):
        _seed_old_soft_deleted_memory(
            conn, "old", days_ago=DEFAULT_RETENTION_DAYS + 10
        )
        result = purge_old_soft_deletes(conn, profile="default", now=NOW)
        assert result.dry_run is False
        assert _count_all_memories(conn) == 0

    def test_retention_result_dry_run_field(self):
        r = RetentionResult(dry_run=True, memories_purged=5)
        assert r.dry_run is True
        assert r.memories_purged == 5
        assert r.total() == 5
