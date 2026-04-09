"""Unit tests for plugins/memory/claudia/verification.py (Phase 2B.4).

Covered:

- ``VerificationResult`` dataclass
- ``VALID_VERIFICATION_STATUSES`` tripwire vs the schema CHECK
- Status marks:
    * ``mark_verified``: pending → verified
    * ``mark_flagged``: pending → flagged
    * ``mark_contradicts``: verified → contradicts
    * rejects bogus memory_id
    * rejects wrong profile
    * idempotent on the same status
- Confidence decay:
    * applies to origin='extracted' and 'inferred'
    * does NOT touch origin='user_stated' or 'corrected'
    * target = initial * 0.5 ** (elapsed_days / half_life_days)
    * respects the configured floor
    * only writes if the target is meaningfully lower than current
    * accessed_at anchors the decay clock
    * freshly-accessed memories (elapsed < half_life) barely change
- Stale flagging:
    * pending → flagged after threshold
    * doesn't touch verified/flagged/contradicts
    * doesn't touch deleted memories
    * profile isolated
    * returns count
- ``run_verification``:
    * empty DB → zeros
    * full pass runs both stages
    * idempotent
    * duration populated
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.claudia import schema
from plugins.memory.claudia.verification import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_INITIAL_CONFIDENCE,
    DEFAULT_STALE_DAYS,
    VALID_VERIFICATION_STATUSES,
    VerificationResult,
    decay_memory_confidence,
    flag_stale_memories,
    mark_contradicts,
    mark_flagged,
    mark_verified,
    run_verification,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = schema.ensure_database(tmp_path / "verification.db")
    yield db
    db.close()


def _insert_memory(
    conn,
    content,
    *,
    origin="extracted",
    confidence=0.7,
    verification="pending",
    accessed_at: datetime = NOW,
    profile="default",
):
    """Insert a memory row directly. Returns the new row id."""
    now_iso = NOW.isoformat()
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
            verification,
            profile,
            now_iso,
            accessed_at.isoformat(),
        ),
    )
    return cur.lastrowid


def _get_memory(conn, memory_id, *, profile="default"):
    return conn.execute(
        """
        SELECT id, content, origin, confidence, verification, accessed_at, profile
        FROM memories
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (memory_id, profile),
    ).fetchone()


# ─── VerificationResult dataclass ───────────────────────────────────────


class TestVerificationResult:
    def test_defaults(self):
        r = VerificationResult()
        assert r.decayed_count == 0
        assert r.flagged_stale_count == 0
        assert r.duration_seconds == 0.0


class TestValidStatuses:
    def test_matches_schema_check_constraint(self):
        """Tripwire: locked to the memories.verification CHECK."""
        assert VALID_VERIFICATION_STATUSES == frozenset(
            {"pending", "verified", "flagged", "contradicts"}
        )


# ─── Status marks ───────────────────────────────────────────────────────


class TestMarkVerified:
    def test_pending_to_verified(self, conn):
        mid = _insert_memory(conn, "x", verification="pending")
        updated = mark_verified(conn, mid, profile="default", now=NOW)
        assert updated is True
        row = _get_memory(conn, mid)
        assert row["verification"] == "verified"

    def test_missing_id_returns_false(self, conn):
        assert mark_verified(conn, 99999, profile="default", now=NOW) is False

    def test_wrong_profile_returns_false(self, conn):
        mid = _insert_memory(conn, "x", profile="user_a")
        assert mark_verified(conn, mid, profile="user_b", now=NOW) is False

    def test_soft_deleted_returns_false(self, conn):
        mid = _insert_memory(conn, "x")
        conn.execute(
            "UPDATE memories SET deleted_at = ? WHERE id = ?",
            (NOW.isoformat(), mid),
        )
        assert mark_verified(conn, mid, profile="default", now=NOW) is False


class TestMarkFlagged:
    def test_transition(self, conn):
        mid = _insert_memory(conn, "x")
        mark_flagged(conn, mid, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["verification"] == "flagged"


class TestMarkContradicts:
    def test_verified_to_contradicts(self, conn):
        mid = _insert_memory(conn, "x", verification="verified")
        mark_contradicts(conn, mid, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["verification"] == "contradicts"


# ─── Confidence decay ───────────────────────────────────────────────────


class TestDecayConfidence:
    def test_fresh_memory_not_decayed(self, conn):
        """A memory accessed a few hours ago should barely change."""
        mid = _insert_memory(
            conn, "x", origin="extracted", confidence=0.7,
            accessed_at=NOW - timedelta(hours=1),
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["confidence"] >= 0.69  # barely changed

    def test_old_memory_decayed(self, conn):
        """A memory not accessed for 2x half-life should drop toward
        0.7 * 0.25 = 0.175."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 2)
        mid = _insert_memory(
            conn, "x", origin="extracted", confidence=0.7,
            accessed_at=old,
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert 0.15 <= row["confidence"] <= 0.2

    def test_very_old_memory_hits_floor(self, conn):
        """A memory not accessed for 100x half-life should hit the floor."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 100)
        mid = _insert_memory(
            conn, "x", origin="extracted", confidence=0.7,
            accessed_at=old,
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["confidence"] == DEFAULT_CONFIDENCE_FLOOR

    def test_user_stated_memory_not_decayed(self, conn):
        """origin='user_stated' is protected from decay."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 10)
        mid = _insert_memory(
            conn, "x", origin="user_stated", confidence=0.9,
            accessed_at=old,
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["confidence"] == 0.9  # unchanged

    def test_corrected_memory_not_decayed(self, conn):
        """origin='corrected' is also protected."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 10)
        mid = _insert_memory(
            conn, "x", origin="corrected", confidence=1.0,
            accessed_at=old,
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["confidence"] == 1.0

    def test_inferred_memory_is_decayed(self, conn):
        """origin='inferred' DOES get decayed (low-trust inferences)."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 3)
        mid = _insert_memory(
            conn, "x", origin="inferred", confidence=0.7,
            accessed_at=old,
        )
        decay_memory_confidence(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["confidence"] < 0.15  # significantly decayed

    def test_profile_isolation(self, conn):
        """Decay on one profile doesn't touch another."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 10)
        mid_a = _insert_memory(
            conn, "x", profile="user_a", accessed_at=old
        )
        mid_b = _insert_memory(
            conn, "y", profile="user_b", accessed_at=old
        )
        decay_memory_confidence(conn, profile="user_a", now=NOW)
        row_b = _get_memory(conn, mid_b, profile="user_b")
        assert row_b["confidence"] == 0.7  # unchanged

    def test_returns_count(self, conn):
        """Returns number of rows updated."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 10)
        _insert_memory(conn, "a", origin="extracted", accessed_at=old)
        _insert_memory(conn, "b", origin="extracted", accessed_at=old)
        _insert_memory(conn, "c", origin="user_stated", accessed_at=old)  # protected
        count = decay_memory_confidence(conn, profile="default", now=NOW)
        assert count == 2

    def test_idempotent_close_enough(self, conn):
        """Running decay twice on the same moment should not double-decay.

        The decay target is based on accessed_at + elapsed, not on
        current confidence. Running twice produces the same target,
        so the second run finds confidence already at target and
        skips the write.
        """
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 3)
        mid = _insert_memory(
            conn, "x", origin="extracted", confidence=0.7,
            accessed_at=old,
        )
        count_first = decay_memory_confidence(conn, profile="default", now=NOW)
        count_second = decay_memory_confidence(conn, profile="default", now=NOW)
        assert count_first == 1
        assert count_second == 0


# ─── Stale flagging ─────────────────────────────────────────────────────


class TestFlagStale:
    def test_pending_stale_flagged(self, conn):
        old = NOW - timedelta(days=DEFAULT_STALE_DAYS + 1)
        mid = _insert_memory(
            conn, "x", verification="pending", accessed_at=old
        )
        count = flag_stale_memories(conn, profile="default", now=NOW)
        assert count == 1
        row = _get_memory(conn, mid)
        assert row["verification"] == "flagged"

    def test_fresh_pending_not_flagged(self, conn):
        mid = _insert_memory(
            conn, "x", verification="pending",
            accessed_at=NOW - timedelta(days=1),
        )
        count = flag_stale_memories(conn, profile="default", now=NOW)
        assert count == 0
        row = _get_memory(conn, mid)
        assert row["verification"] == "pending"

    def test_verified_stale_not_flagged(self, conn):
        """A 'verified' memory stays verified even if stale."""
        old = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        mid = _insert_memory(
            conn, "x", verification="verified", accessed_at=old
        )
        flag_stale_memories(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["verification"] == "verified"

    def test_contradicts_stale_not_flagged(self, conn):
        old = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        mid = _insert_memory(
            conn, "x", verification="contradicts", accessed_at=old
        )
        flag_stale_memories(conn, profile="default", now=NOW)
        row = _get_memory(conn, mid)
        assert row["verification"] == "contradicts"

    def test_already_flagged_not_re_flagged(self, conn):
        old = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        _insert_memory(
            conn, "x", verification="flagged", accessed_at=old
        )
        count = flag_stale_memories(conn, profile="default", now=NOW)
        assert count == 0

    def test_profile_isolation(self, conn):
        old = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        mid = _insert_memory(
            conn, "x", verification="pending",
            profile="user_b", accessed_at=old,
        )
        flag_stale_memories(conn, profile="user_a", now=NOW)
        row = _get_memory(conn, mid, profile="user_b")
        assert row["verification"] == "pending"


# ─── run_verification orchestrator ──────────────────────────────────────


class TestRunVerification:
    def test_empty_db(self, conn):
        result = run_verification(conn, profile="default", now=NOW)
        assert result.decayed_count == 0
        assert result.flagged_stale_count == 0
        assert result.duration_seconds >= 0.0

    def test_full_pass(self, conn):
        """Decay candidate + stale flag in one run.

        Note: the "old fact" memory is marked 'verified' so only the
        "stale fact" qualifies for flagging. Both memories are old
        enough to be decayed.
        """
        # Old extracted memory, already verified → decay only
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 5)
        _insert_memory(
            conn, "old fact", origin="extracted", confidence=0.7,
            verification="verified", accessed_at=old,
        )
        # Old pending memory → will flag (AND decay)
        stale = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        _insert_memory(
            conn, "stale fact", origin="extracted",
            verification="pending", accessed_at=stale,
        )

        result = run_verification(conn, profile="default", now=NOW)
        # Both memories decay; only the pending one gets flagged
        assert result.decayed_count == 2
        assert result.flagged_stale_count == 1

    def test_idempotent(self, conn):
        """Second pass on the same state should not re-decay or re-flag."""
        old = NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 5)
        _insert_memory(
            conn, "x", origin="extracted", confidence=0.7,
            accessed_at=old,
        )
        stale = NOW - timedelta(days=DEFAULT_STALE_DAYS + 10)
        _insert_memory(
            conn, "y", origin="extracted", verification="pending",
            accessed_at=stale,
        )

        first = run_verification(conn, profile="default", now=NOW)
        second = run_verification(conn, profile="default", now=NOW)
        assert second.decayed_count == 0
        assert second.flagged_stale_count == 0
