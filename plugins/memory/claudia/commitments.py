"""Commitment CRUD for the Claudia memory provider (Phase 2B.2).

The ``commitments`` table was defined in Phase 2A.2a (schema.py) but
had no read/write layer. This module supplies that layer in the same
shape as ``entities.py``: a dataclass, CRUD helpers, and stateless
read semantics.

Design principles (mirror entities.py so the layers feel consistent):

- **Profile isolation is mandatory.** Every function takes a
  ``profile`` parameter and every query filters on it. Two users
  can have the same commitment text without collision.

- **Soft deletes only.** ``deleted_at`` timestamp; every read
  filters ``WHERE deleted_at IS NULL``.

- **Stateless reads.** ``get_commitment``, ``find_commitment_by_content``,
  and ``list_commitments`` do NOT bump access counts or touch
  ``updated_at``. Writes (status changes) use the writer queue.

- **CHECK constraint mirrored in Python.** ``VALID_COMMITMENT_STATUSES``
  duplicates the schema's ``status IN (...)`` check. A tripwire
  test locks them together.

- **Foreign keys are optional.** ``owner_entity_id`` and
  ``target_entity_id`` are nullable and default to None. The
  commitment detector (Phase 2B.2 wire-up) writes commitments
  before entity extraction has resolved names to IDs, so the FKs
  get populated later by a consolidation pass (Phase 2B.3). The
  ``content`` column always carries the full action text, so
  queries work even with NULL FKs.

- **completed_at is write-once-on-status-change.** Moving status
  from anything to ``completed`` sets ``completed_at`` to the
  current time. Moving from ``completed`` to anything else leaves
  ``completed_at`` alone (historical record — you can check
  ``status=='completed'`` for "currently done" and
  ``completed_at IS NOT NULL`` for "was ever done").

Public API:

- ``Commitment`` dataclass
- ``VALID_COMMITMENT_STATUSES`` frozenset
- ``create_commitment(conn, content, **kwargs) -> Commitment``
- ``get_commitment(conn, commitment_id, *, profile) -> Optional[Commitment]``
- ``list_commitments(conn, *, status, profile, limit) -> List[Commitment]``
- ``find_commitment_by_content(conn, content, *, profile) -> Optional[Commitment]``
- ``update_commitment_status(conn, commitment_id, new_status, *, profile, now) -> Optional[Commitment]``
- ``soft_delete_commitment(conn, commitment_id, *, profile, now) -> bool``

Reference: docs/decisions/memory-provider-design.md and
plugins/memory/claudia/schema.py (commitments table DDL)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────


#: Valid commitment statuses, matching the CHECK constraint in
#: schema.py. A tripwire test in test_commitments.py locks the
#: set — if you add a status here, add it to schema.py (as a
#: migration, not an ALTER) and update the tripwire.
VALID_COMMITMENT_STATUSES = frozenset({
    "open",
    "completed",
    "overdue",
    "dropped",
    "superseded",
})


# ─── Dataclass ──────────────────────────────────────────────────────────


@dataclass
class Commitment:
    """A single commitments table row.

    All fields are populated from the DB row. FK fields
    (``owner_entity_id``, ``target_entity_id``) are typed as
    ``Optional[int]`` to reflect the nullable columns.
    """

    id: int
    content: str
    owner_entity_id: Optional[int] = None
    target_entity_id: Optional[int] = None
    deadline: Optional[str] = None
    status: str = "open"
    source_type: Optional[str] = None
    source_ref: Optional[str] = None
    profile: str = "default"
    created_at: str = ""
    updated_at: str = ""
    completed_at: Optional[str] = None


# ─── Helpers ────────────────────────────────────────────────────────────


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def _validate_status(status: str) -> None:
    if status not in VALID_COMMITMENT_STATUSES:
        raise ValueError(
            f"invalid commitment status: {status!r}. "
            f"Must be one of {sorted(VALID_COMMITMENT_STATUSES)}"
        )


def _row_to_commitment(row: sqlite3.Row) -> Commitment:
    return Commitment(
        id=int(row["id"]),
        content=row["content"],
        owner_entity_id=row["owner_entity_id"],
        target_entity_id=row["target_entity_id"],
        deadline=row["deadline"],
        status=row["status"],
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        profile=row["profile"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


# ─── create_commitment ──────────────────────────────────────────────────


def create_commitment(
    conn: sqlite3.Connection,
    content: str,
    *,
    owner_entity_id: Optional[int] = None,
    target_entity_id: Optional[int] = None,
    deadline: Optional[str] = None,
    status: str = "open",
    source_type: Optional[str] = None,
    source_ref: Optional[str] = None,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Commitment:
    """Insert a new commitment row and return the populated dataclass.

    Raises:
        ValueError: if ``status`` is not in VALID_COMMITMENT_STATUSES.
        sqlite3.IntegrityError: on CHECK constraint failure (should
            not happen because we pre-validate ``status``).
    """
    _validate_status(status)
    now_iso = _now_iso(now)

    cur = conn.execute(
        """
        INSERT INTO commitments (
            content, owner_entity_id, target_entity_id, deadline,
            status, source_type, source_ref, profile,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            owner_entity_id,
            target_entity_id,
            deadline,
            status,
            source_type,
            source_ref,
            profile,
            now_iso,
            now_iso,
        ),
    )
    commitment_id = cur.lastrowid

    fetched = get_commitment(conn, commitment_id, profile=profile)
    if fetched is None:  # pragma: no cover - should never happen
        raise RuntimeError(
            f"commitment {commitment_id} vanished after insert"
        )
    return fetched


# ─── get_commitment ─────────────────────────────────────────────────────


def get_commitment(
    conn: sqlite3.Connection,
    commitment_id: int,
    *,
    profile: str = "default",
) -> Optional[Commitment]:
    """Return a commitment by primary key, or None if missing/deleted/wrong profile."""
    row = conn.execute(
        """
        SELECT id, content, owner_entity_id, target_entity_id,
               deadline, status, source_type, source_ref, profile,
               created_at, updated_at, completed_at
        FROM commitments
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (commitment_id, profile),
    ).fetchone()

    if row is None:
        return None
    return _row_to_commitment(row)


# ─── list_commitments ───────────────────────────────────────────────────


def list_commitments(
    conn: sqlite3.Connection,
    *,
    status: Optional[str] = None,
    profile: str = "default",
    limit: int = 100,
) -> List[Commitment]:
    """Return commitments, ordered by deadline ASC (NULLs last), then created_at DESC.

    ``status`` optionally restricts to a single status. ``limit``
    caps the row count (default 100, enough for interactive use).
    Soft-deleted rows are always excluded.
    """
    if status is not None:
        _validate_status(status)

    sql = [
        "SELECT id, content, owner_entity_id, target_entity_id,",
        "       deadline, status, source_type, source_ref, profile,",
        "       created_at, updated_at, completed_at",
        "FROM commitments",
        "WHERE profile = ? AND deleted_at IS NULL",
    ]
    params: List = [profile]

    if status is not None:
        sql.append("AND status = ?")
        params.append(status)

    # NULLs-last ordering: put rows with a deadline first (sorted
    # ASC so earliest is first), then rows without a deadline,
    # then tie-break on created_at DESC so newer rows win within
    # the same deadline bucket.
    sql.append("ORDER BY (deadline IS NULL), deadline ASC, created_at DESC")
    sql.append("LIMIT ?")
    params.append(limit)

    rows = conn.execute("\n".join(sql), params).fetchall()
    return [_row_to_commitment(row) for row in rows]


# ─── find_commitment_by_content ─────────────────────────────────────────


def find_commitment_by_content(
    conn: sqlite3.Connection,
    content: str,
    *,
    profile: str = "default",
) -> Optional[Commitment]:
    """Resolve a commitment by exact (case-insensitive) content match.

    Used by the detector for idempotency checks: if a user says
    "I'll send the proposal" in two turns, the detector should
    update the existing row rather than create a duplicate (future
    sub-task — Phase 2B.3 handles fuzzy dedup). For now this is
    exact-match only.

    On duplicates, returns the most recently-created row so a
    commitment that was marked ``completed`` and then mentioned
    again is still surface-able.
    """
    if not content:
        return None

    row = conn.execute(
        """
        SELECT id, content, owner_entity_id, target_entity_id,
               deadline, status, source_type, source_ref, profile,
               created_at, updated_at, completed_at
        FROM commitments
        WHERE profile = ?
          AND deleted_at IS NULL
          AND LOWER(content) = LOWER(?)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (profile, content),
    ).fetchone()

    if row is None:
        return None
    return _row_to_commitment(row)


# ─── update_commitment_status ───────────────────────────────────────────


def update_commitment_status(
    conn: sqlite3.Connection,
    commitment_id: int,
    new_status: str,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Optional[Commitment]:
    """Transition a commitment to a new status.

    Side effects:
    - ``updated_at`` is always refreshed to ``now``.
    - Moving to ``completed`` sets ``completed_at`` to ``now`` if
      it wasn't already set.
    - Moving from ``completed`` to anything else leaves
      ``completed_at`` alone (historical record).

    Returns the updated Commitment, or None if the row doesn't
    exist, is soft-deleted, or belongs to another profile.

    Raises:
        ValueError: if ``new_status`` is not valid.
    """
    _validate_status(new_status)
    now_iso = _now_iso(now)

    # Pre-check existence so we can return None cleanly (rather
    # than succeeding with zero rows affected).
    existing = get_commitment(conn, commitment_id, profile=profile)
    if existing is None:
        return None

    if new_status == "completed" and existing.completed_at is None:
        conn.execute(
            """
            UPDATE commitments
            SET status = ?, updated_at = ?, completed_at = ?
            WHERE id = ? AND profile = ? AND deleted_at IS NULL
            """,
            (new_status, now_iso, now_iso, commitment_id, profile),
        )
    else:
        conn.execute(
            """
            UPDATE commitments
            SET status = ?, updated_at = ?
            WHERE id = ? AND profile = ? AND deleted_at IS NULL
            """,
            (new_status, now_iso, commitment_id, profile),
        )

    return get_commitment(conn, commitment_id, profile=profile)


# ─── soft_delete_commitment ─────────────────────────────────────────────


def soft_delete_commitment(
    conn: sqlite3.Connection,
    commitment_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark a commitment as deleted. Returns True on success.

    Idempotent: calling on an already-deleted row (or a row in
    another profile, or a missing row) returns False without
    raising. Matches the pattern of ``entities.soft_delete_entity``.
    """
    now_iso = _now_iso(now)
    cur = conn.execute(
        """
        UPDATE commitments
        SET deleted_at = ?, updated_at = ?
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (now_iso, now_iso, commitment_id, profile),
    )
    return cur.rowcount > 0
