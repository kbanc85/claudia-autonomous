"""Unit tests for plugins/memory/claudia/commitments.py (Phase 2B.2).

The commitments table was defined in Phase 2A.2a schema.py but had
no CRUD layer. Phase 2B.2 adds that layer. This file exercises
every public function in commitments.py against a real SQLite
database created via ``schema.ensure_database``.

Covered:

- ``Commitment`` dataclass shape and defaults
- ``create_commitment``:
    * minimal args (content only)
    * all fields populated
    * CHECK constraint on status
    * profile isolation
    * provenance preservation (source_type, source_ref)
- ``get_commitment``:
    * returns None on missing id
    * returns None on wrong profile
    * returns None on soft-deleted row
- ``list_commitments``:
    * empty list on empty table
    * filter by status
    * filter by profile
    * order by deadline (NULLs last)
    * excludes soft-deleted
- ``find_commitment_by_content``:
    * exact match wins
    * case insensitive
    * returns the most recent on duplicates
    * excludes soft-deleted
- ``update_commitment_status``:
    * open → completed sets completed_at
    * open → overdue leaves completed_at null
    * rejects invalid status
    * updates updated_at
    * returns None on missing row
- ``soft_delete_commitment``:
    * sets deleted_at
    * subsequent get returns None
    * subsequent list_commitments excludes row
- Schema compatibility:
    * CHECK constraint rejects bogus status
    * FK fields nullable (owner_entity_id, target_entity_id)

Tests follow the Phase 2A conventions:
- Deterministic ``NOW`` datetime anchor
- Real SQLite via ``schema.ensure_database(tmp_path / "x.db")``
- One behavior per test, class-grouped by concern
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.claudia import schema
from plugins.memory.claudia.commitments import (
    VALID_COMMITMENT_STATUSES,
    Commitment,
    create_commitment,
    find_commitment_by_content,
    get_commitment,
    list_commitments,
    soft_delete_commitment,
    update_commitment_status,
)


#: Deterministic time anchor. All tests that care about timestamps
#: use this so assertions are reproducible.
NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh migrated database, unique per test."""
    db = schema.ensure_database(tmp_path / "commitments.db")
    yield db
    db.close()


# ─── Dataclass shape ────────────────────────────────────────────────────


class TestCommitmentDataclass:
    def test_defaults_populated(self):
        c = Commitment(id=1, content="do something")
        assert c.owner_entity_id is None
        assert c.target_entity_id is None
        assert c.deadline is None
        assert c.status == "open"
        assert c.source_type is None
        assert c.source_ref is None
        assert c.profile == "default"
        assert c.completed_at is None

    def test_all_fields_settable(self):
        c = Commitment(
            id=42,
            content="send proposal",
            owner_entity_id=1,
            target_entity_id=2,
            deadline="2026-04-12T17:00:00+00:00",
            status="completed",
            source_type="conversation",
            source_ref="sess-abc",
            profile="user_123",
            created_at="2026-04-09T12:00:00+00:00",
            updated_at="2026-04-09T12:00:00+00:00",
            completed_at="2026-04-09T18:00:00+00:00",
        )
        assert c.id == 42
        assert c.status == "completed"
        assert c.profile == "user_123"

    def test_valid_statuses_matches_schema(self):
        """VALID_COMMITMENT_STATUSES must match the schema CHECK constraint.

        Tripwire: if these drift, create_commitment will accept values
        that SQLite rejects at insert time, producing confusing errors.
        The schema CHECK is: ('open', 'completed', 'overdue',
        'dropped', 'superseded').
        """
        assert VALID_COMMITMENT_STATUSES == frozenset(
            {"open", "completed", "overdue", "dropped", "superseded"}
        )


# ─── create_commitment ──────────────────────────────────────────────────


class TestCreateCommitment:
    def test_minimal_args(self, conn):
        c = create_commitment(conn, "send the proposal", now=NOW)
        assert c.id is not None
        assert c.content == "send the proposal"
        assert c.status == "open"
        assert c.owner_entity_id is None
        assert c.target_entity_id is None
        assert c.deadline is None
        assert c.profile == "default"
        assert c.created_at == NOW.isoformat()
        assert c.updated_at == NOW.isoformat()
        assert c.completed_at is None

    def test_all_fields(self, conn):
        c = create_commitment(
            conn,
            "schedule the kickoff",
            deadline="2026-04-15",
            status="open",
            source_type="conversation",
            source_ref="sess-42",
            profile="user_a",
            now=NOW,
        )
        assert c.deadline == "2026-04-15"
        assert c.source_type == "conversation"
        assert c.source_ref == "sess-42"
        assert c.profile == "user_a"

    def test_default_status_is_open(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        assert c.status == "open"

    def test_rejects_invalid_status(self, conn):
        with pytest.raises(ValueError):
            create_commitment(conn, "x", status="bogus", now=NOW)

    def test_accepts_all_valid_statuses(self, conn):
        for i, s in enumerate(sorted(VALID_COMMITMENT_STATUSES)):
            c = create_commitment(conn, f"task {i}", status=s, now=NOW)
            assert c.status == s

    def test_profile_isolation_same_content(self, conn):
        """Same content in two profiles creates two separate rows."""
        a = create_commitment(conn, "x", profile="user_a", now=NOW)
        b = create_commitment(conn, "x", profile="user_b", now=NOW)
        assert a.id != b.id

    def test_timestamps_match_now(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        assert c.created_at == NOW.isoformat()
        assert c.updated_at == NOW.isoformat()

    def test_returns_populated_dataclass(self, conn):
        c = create_commitment(conn, "x", source_type="capture", now=NOW)
        assert isinstance(c, Commitment)
        assert c.source_type == "capture"


# ─── get_commitment ─────────────────────────────────────────────────────


class TestGetCommitment:
    def test_round_trip(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        fetched = get_commitment(conn, c.id)
        assert fetched is not None
        assert fetched.id == c.id
        assert fetched.content == "x"

    def test_missing_id_returns_none(self, conn):
        assert get_commitment(conn, 99999) is None

    def test_wrong_profile_returns_none(self, conn):
        c = create_commitment(conn, "x", profile="user_a", now=NOW)
        assert get_commitment(conn, c.id, profile="user_b") is None

    def test_correct_profile_returns_row(self, conn):
        c = create_commitment(conn, "x", profile="user_a", now=NOW)
        assert get_commitment(conn, c.id, profile="user_a") is not None

    def test_soft_deleted_returns_none(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        soft_delete_commitment(conn, c.id, now=NOW)
        assert get_commitment(conn, c.id) is None


# ─── list_commitments ───────────────────────────────────────────────────


class TestListCommitments:
    def test_empty_table(self, conn):
        assert list_commitments(conn) == []

    def test_returns_all_on_no_filter(self, conn):
        create_commitment(conn, "a", now=NOW)
        create_commitment(conn, "b", now=NOW)
        create_commitment(conn, "c", now=NOW)
        rows = list_commitments(conn)
        assert len(rows) == 3

    def test_filter_by_status(self, conn):
        create_commitment(conn, "a", status="open", now=NOW)
        create_commitment(conn, "b", status="completed", now=NOW)
        create_commitment(conn, "c", status="open", now=NOW)
        open_rows = list_commitments(conn, status="open")
        completed_rows = list_commitments(conn, status="completed")
        assert len(open_rows) == 2
        assert len(completed_rows) == 1

    def test_filter_by_profile(self, conn):
        create_commitment(conn, "a", profile="user_a", now=NOW)
        create_commitment(conn, "b", profile="user_b", now=NOW)
        assert len(list_commitments(conn, profile="user_a")) == 1
        assert len(list_commitments(conn, profile="user_b")) == 1

    def test_excludes_soft_deleted(self, conn):
        a = create_commitment(conn, "a", now=NOW)
        create_commitment(conn, "b", now=NOW)
        soft_delete_commitment(conn, a.id, now=NOW)
        rows = list_commitments(conn)
        assert len(rows) == 1
        assert rows[0].content == "b"

    def test_order_by_deadline_nulls_last(self, conn):
        """Rows with a deadline come before rows without one; within
        those with deadlines, earlier deadlines come first."""
        create_commitment(conn, "no deadline", deadline=None, now=NOW)
        create_commitment(conn, "later", deadline="2026-05-01", now=NOW)
        create_commitment(conn, "sooner", deadline="2026-04-10", now=NOW)
        rows = list_commitments(conn)
        assert rows[0].content == "sooner"
        assert rows[1].content == "later"
        assert rows[2].content == "no deadline"

    def test_limit(self, conn):
        for i in range(5):
            create_commitment(conn, f"task {i}", now=NOW)
        rows = list_commitments(conn, limit=3)
        assert len(rows) == 3


# ─── find_commitment_by_content ─────────────────────────────────────────


class TestFindCommitmentByContent:
    def test_exact_match(self, conn):
        c = create_commitment(conn, "send the proposal", now=NOW)
        found = find_commitment_by_content(conn, "send the proposal")
        assert found is not None
        assert found.id == c.id

    def test_case_insensitive(self, conn):
        c = create_commitment(conn, "Send The Proposal", now=NOW)
        found = find_commitment_by_content(conn, "send the proposal")
        assert found is not None
        assert found.id == c.id

    def test_no_match_returns_none(self, conn):
        create_commitment(conn, "a", now=NOW)
        assert find_commitment_by_content(conn, "b") is None

    def test_profile_isolated(self, conn):
        create_commitment(conn, "shared text", profile="user_a", now=NOW)
        assert (
            find_commitment_by_content(conn, "shared text", profile="user_b")
            is None
        )

    def test_excludes_soft_deleted(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        soft_delete_commitment(conn, c.id, now=NOW)
        assert find_commitment_by_content(conn, "x") is None

    def test_most_recent_wins_on_duplicates(self, conn):
        """When two rows share the same content, return the newest.

        This is used by the detector for idempotency checks: if a
        commitment is detected again, we want to find the latest
        record (which might be in a different status).
        """
        older = datetime(2026, 4, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 4, 5, tzinfo=timezone.utc)
        create_commitment(conn, "x", now=older)
        c2 = create_commitment(conn, "x", now=newer)
        found = find_commitment_by_content(conn, "x")
        assert found is not None
        assert found.id == c2.id


# ─── update_commitment_status ───────────────────────────────────────────


class TestUpdateCommitmentStatus:
    def test_open_to_completed_sets_completed_at(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        later = NOW + timedelta(hours=6)
        updated = update_commitment_status(
            conn, c.id, "completed", now=later
        )
        assert updated is not None
        assert updated.status == "completed"
        assert updated.completed_at == later.isoformat()

    def test_open_to_overdue_leaves_completed_at_null(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        later = NOW + timedelta(days=7)
        updated = update_commitment_status(conn, c.id, "overdue", now=later)
        assert updated is not None
        assert updated.status == "overdue"
        assert updated.completed_at is None

    def test_updates_updated_at(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        later = NOW + timedelta(hours=1)
        updated = update_commitment_status(conn, c.id, "dropped", now=later)
        assert updated.updated_at == later.isoformat()

    def test_rejects_invalid_status(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        with pytest.raises(ValueError):
            update_commitment_status(conn, c.id, "nonsense", now=NOW)

    def test_missing_id_returns_none(self, conn):
        assert (
            update_commitment_status(conn, 99999, "completed", now=NOW)
            is None
        )

    def test_soft_deleted_row_not_updated(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        soft_delete_commitment(conn, c.id, now=NOW)
        result = update_commitment_status(conn, c.id, "completed", now=NOW)
        assert result is None


# ─── soft_delete_commitment ─────────────────────────────────────────────


class TestSoftDeleteCommitment:
    def test_sets_deleted_at(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        later = NOW + timedelta(minutes=5)
        assert soft_delete_commitment(conn, c.id, now=later) is True
        # Raw SQL check — the dataclass path filters deleted_at
        row = conn.execute(
            "SELECT deleted_at FROM commitments WHERE id = ?", (c.id,)
        ).fetchone()
        assert row["deleted_at"] == later.isoformat()

    def test_missing_id_returns_false(self, conn):
        assert soft_delete_commitment(conn, 99999, now=NOW) is False

    def test_idempotent_on_already_deleted(self, conn):
        c = create_commitment(conn, "x", now=NOW)
        soft_delete_commitment(conn, c.id, now=NOW)
        # Re-deleting is still a no-op success (filter excludes deleted)
        result = soft_delete_commitment(conn, c.id, now=NOW)
        assert result is False


# ─── Schema compatibility ──────────────────────────────────────────────


class TestSchemaCompatibility:
    def test_check_constraint_rejects_bogus_status_at_sqlite_level(self, conn):
        """Even if someone bypasses create_commitment, SQLite itself
        rejects invalid status via the CHECK constraint."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO commitments
                    (content, status, profile, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("x", "invalid", "default", NOW.isoformat(), NOW.isoformat()),
            )

    def test_fk_fields_nullable(self, conn):
        """owner_entity_id and target_entity_id can both be NULL.

        The Phase 2B.2 detector relies on this because extraction
        runs concurrently with detection — entity IDs aren't yet
        resolved when the commitment is written. The FKs get
        populated later by consolidation (Phase 2B.3).
        """
        c = create_commitment(conn, "x", now=NOW)
        assert c.owner_entity_id is None
        assert c.target_entity_id is None

    def test_fk_fields_preserved_when_set(self, conn):
        """When real entity IDs are provided, FK fields round-trip.

        Creates two entities first because the schema has
        ``PRAGMA foreign_keys=ON`` — passing bogus IDs would raise
        IntegrityError. This test also exercises the "eventually
        linked" path that consolidation will use.
        """
        from plugins.memory.claudia import entities

        owner = entities.create_entity(
            conn, "person", "User Self", now=NOW
        )
        target = entities.create_entity(
            conn, "person", "Sarah Chen", now=NOW
        )

        c = create_commitment(
            conn,
            "send Sarah the proposal",
            owner_entity_id=owner.id,
            target_entity_id=target.id,
            now=NOW,
        )
        fetched = get_commitment(conn, c.id)
        assert fetched.owner_entity_id == owner.id
        assert fetched.target_entity_id == target.id

    def test_fk_integrity_rejects_bogus_ids(self, conn):
        """Schema has PRAGMA foreign_keys=ON so invalid FKs raise.

        Tripwire: if someone disables the pragma in the future,
        this test will start passing (silent data corruption).
        """
        with pytest.raises(sqlite3.IntegrityError):
            create_commitment(
                conn,
                "x",
                owner_entity_id=99999,
                target_entity_id=99999,
                now=NOW,
            )
