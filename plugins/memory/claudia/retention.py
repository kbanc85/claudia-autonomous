"""Retention / purge for Claudia memory (Phase 2C.4).

Permanent removal of soft-deleted rows older than a configurable
retention window. Soft-deletes accumulate indefinitely without
cleanup — this module bounds the growth.

Scope: operates on all four soft-delete-bearing tables in one
pass — ``memories``, ``entities``, ``relationships``,
``commitments``. For each, rows with
``deleted_at IS NOT NULL AND deleted_at < cutoff`` are
permanently removed via an actual ``DELETE`` statement (unlike
``soft_delete_*`` which just sets ``deleted_at``).

Design principles:

- **Separate from consolidation.** Retention purge is destructive
  and irreversible. It deserves its own explicit entry point
  (``provider.purge_old_soft_deletes``) rather than being
  bundled into ``consolidate()``. Users call it deliberately,
  on a longer cadence than consolidation.

- **Profile isolation.** All deletes filter on ``profile``.
  Purging one user's data never touches another's.

- **Retention window is a parameter.** The default (90 days)
  is long enough that honest mistakes (accidental merges, bad
  extractions) are still recoverable — but short enough that a
  year of churn doesn't fill the disk. Callers can override.

- **Returns counts, not IDs.** ``RetentionResult`` reports how
  many rows were purged per table but not WHICH rows. Once
  they're gone, they're gone — we don't retain audit trails for
  audited deletions. The soft-delete timestamp was the audit
  trail.

Public API:

- ``RetentionResult`` dataclass (per-table counts, .total() helper)
- ``DEFAULT_RETENTION_DAYS = 90``
- ``purge_old_soft_deletes(conn, *, profile, now, retention_days)``
  → RetentionResult

Called from ``ClaudiaMemoryProvider.purge_old_soft_deletes()``
via the writer queue so the purge commits inside the writer's
transaction and interleaves cleanly with normal writes.

Reference: plans/phase-2b-handoff.md Phase 2C scope notes.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


#: Default retention window: ninety days. Long enough for realistic
#: "wait, that was wrong" recovery windows, short enough to keep
#: the soft-delete pile bounded on long-running sessions.
DEFAULT_RETENTION_DAYS = 90

#: Tables that carry a ``deleted_at`` column. Any new soft-delete-
#: bearing table must be added here for retention to cover it.
_SOFT_DELETE_TABLES = (
    "memories",
    "entities",
    "relationships",
    "commitments",
)


@dataclass
class RetentionResult:
    """Summary of a single retention purge pass.

    Per-table counts let callers log which tables dominated the
    purge. The ``total()`` helper is for quick reporting.

    When ``dry_run`` is True, the counts reflect what WOULD have
    been purged if the pass had committed — no rows were deleted.
    """

    memories_purged: int = 0
    entities_purged: int = 0
    relationships_purged: int = 0
    commitments_purged: int = 0
    dry_run: bool = False

    def total(self) -> int:
        return (
            self.memories_purged
            + self.entities_purged
            + self.relationships_purged
            + self.commitments_purged
        )


def purge_old_soft_deletes(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
) -> RetentionResult:
    """Permanently remove soft-deleted rows older than the retention window.

    For each of the four soft-delete-bearing tables, issues:

        DELETE FROM {table}
        WHERE profile = ?
          AND deleted_at IS NOT NULL
          AND deleted_at < ?

    where the cutoff is ``now - retention_days``. Non-deleted
    rows are never touched; recently-soft-deleted rows (inside
    the window) are preserved so recovery is still possible.

    ``retention_days`` of 0 or less purges all soft-deleted rows
    regardless of age (clamped to 0 for the cutoff math).

    ``dry_run`` (Phase 2D.6): when True, runs SELECT COUNT(*)
    instead of DELETE for each table. The returned counts reflect
    what WOULD have been purged; no rows are removed.

    Returns a ``RetentionResult`` with per-table counts.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if retention_days < 0:
        retention_days = 0

    cutoff = (now - timedelta(days=retention_days)).isoformat()

    counts = {}
    for table in _SOFT_DELETE_TABLES:
        if dry_run:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS n FROM {table}
                WHERE profile = ?
                  AND deleted_at IS NOT NULL
                  AND deleted_at < ?
                """,
                (profile, cutoff),
            ).fetchone()
            counts[table] = row["n"]
        else:
            cur = conn.execute(
                f"""
                DELETE FROM {table}
                WHERE profile = ?
                  AND deleted_at IS NOT NULL
                  AND deleted_at < ?
                """,
                (profile, cutoff),
            )
            counts[table] = cur.rowcount

    return RetentionResult(
        memories_purged=counts.get("memories", 0),
        entities_purged=counts.get("entities", 0),
        relationships_purged=counts.get("relationships", 0),
        commitments_purged=counts.get("commitments", 0),
        dry_run=dry_run,
    )
