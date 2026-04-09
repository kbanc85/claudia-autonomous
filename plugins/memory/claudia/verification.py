"""Verification service for Claudia memory (Phase 2B.4).

Background maintenance for the ``memories`` table:

- **Confidence decay.** Memories with origin ``extracted`` or
  ``inferred`` lose confidence over time. The decay is exponential
  with a configurable half-life (default 30 days) anchored on
  ``accessed_at``. User-stated and user-corrected memories are
  exempt: their confidence reflects explicit human judgment and
  must not be silently lowered.

- **Stale flagging.** Memories still in the ``pending`` verification
  state after ``DEFAULT_STALE_DAYS`` (default 60) are promoted to
  ``flagged``. This is a "hey, I haven't been able to confirm this"
  signal for the user. Verified, flagged, and contradicts memories
  are not touched.

- **Explicit status marks.** ``mark_verified``, ``mark_flagged``,
  and ``mark_contradicts`` change a single memory's verification
  column. Used when the user or the system confirms or disputes
  a specific fact.

Design principles:

- **Idempotent.** Running verification twice on the same state
  produces zeros the second time. Decay targets are absolute
  (derived from ``accessed_at`` and the original-confidence
  constant), not relative to the current value, so re-runs don't
  compound.

- **No cross-profile bleed.** Every query filters on ``profile``.

- **Stateless reads.** The scan phase of ``decay_memory_confidence``
  reads memories but does not bump access_count or accessed_at
  (invariant #18). Writes use a simple UPDATE — no SELECT ... FOR
  UPDATE because the writer queue already serializes access.

- **Protected origins.** ``origin IN ('user_stated', 'corrected')``
  memories are exempt from decay. These represent explicit high-
  confidence assertions from the user (or user corrections) and
  must not drift downward silently. Only ``extracted`` and
  ``inferred`` memories decay — the LLM-derived facts.

Public API:

- ``VerificationResult`` dataclass (decayed_count, flagged_stale_count, duration_seconds)
- ``VALID_VERIFICATION_STATUSES`` frozenset matching schema CHECK
- ``mark_verified(conn, memory_id, *, profile, now)``
- ``mark_flagged(conn, memory_id, *, profile, now)``
- ``mark_contradicts(conn, memory_id, *, profile, now)``
- ``decay_memory_confidence(conn, *, profile, now, half_life_days, floor)``
- ``flag_stale_memories(conn, *, profile, now, stale_days)``
- ``run_verification(conn, *, profile, now)`` → VerificationResult

Called from ``ClaudiaMemoryProvider.verify()`` via the writer queue
so the whole pass commits atomically alongside normal writes.

Reference: plugins/memory/claudia/schema.py for the memories
table DDL (confidence, verification, accessed_at columns),
plans/phase-2b-handoff.md for Phase 2B.4 scope notes.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────


#: Valid values for memories.verification. Locked to the schema
#: CHECK constraint via a tripwire test.
VALID_VERIFICATION_STATUSES = frozenset({
    "pending",
    "verified",
    "flagged",
    "contradicts",
})

#: Memory origin values that are eligible for confidence decay.
#: user_stated and corrected memories are explicit human assertions
#: and must not be silently lowered.
_DECAYABLE_ORIGINS = ("extracted", "inferred")

#: Default half-life in days. Memories accessed this long ago have
#: their confidence halved. Chosen to balance "keep fresh facts"
#: (short half-life = aggressive decay) against "don't lose
#: everything" (long half-life = minimal decay). 30 days matches a
#: typical "monthly review" cadence.
DEFAULT_HALF_LIFE_DAYS = 30

#: Default floor. Confidence cannot decay below this value. A floor
#: of 0.1 means a memory is never considered completely worthless —
#: it can still surface in recall, just with minimal weight.
DEFAULT_CONFIDENCE_FLOOR = 0.1

#: Default initial confidence — the schema's DEFAULT for extracted
#: memories. Decay targets start from this value, not the memory's
#: current stored confidence. This keeps the decay function
#: idempotent (re-runs don't compound).
DEFAULT_INITIAL_CONFIDENCE = 0.7

#: Default days before a pending memory is auto-flagged as stale.
#: 60 days means "if you haven't verified this in two months, I'm
#: going to start doubting it".
DEFAULT_STALE_DAYS = 60

#: A decay write is only issued if the target confidence is this
#: much below the current stored value. Prevents pointless writes
#: for tiny floating-point differences.
_DECAY_WRITE_EPSILON = 0.005


# ─── Dataclass ──────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Summary of a single verification pass.

    decayed_count: how many memory rows had their confidence lowered.
    flagged_stale_count: how many pending memories were flagged.
    duration_seconds: wall-clock time for the pass.
    """

    decayed_count: int = 0
    flagged_stale_count: int = 0
    duration_seconds: float = 0.0


# ─── Helpers ────────────────────────────────────────────────────────────


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def _parse_accessed_at(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 accessed_at string, tolerating tz variants."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_target_confidence(
    elapsed_days: float,
    *,
    initial: float,
    half_life_days: float,
    floor: float,
) -> float:
    """Apply exponential decay for ``elapsed_days`` past ``accessed_at``.

    Target = max(floor, initial * 2**(-elapsed/half_life))
    """
    if elapsed_days <= 0:
        return initial
    periods = elapsed_days / half_life_days
    decayed = initial * (0.5 ** periods)
    return max(floor, decayed)


# ─── Status marks ───────────────────────────────────────────────────────


def _mark_status(
    conn: sqlite3.Connection,
    memory_id: int,
    new_status: str,
    *,
    profile: str,
    now: Optional[datetime],
) -> bool:
    if new_status not in VALID_VERIFICATION_STATUSES:
        raise ValueError(
            f"invalid verification status: {new_status!r}. "
            f"Must be one of {sorted(VALID_VERIFICATION_STATUSES)}"
        )

    cur = conn.execute(
        """
        UPDATE memories
        SET verification = ?
        WHERE id = ?
          AND profile = ?
          AND deleted_at IS NULL
        """,
        (new_status, memory_id, profile),
    )
    return cur.rowcount > 0


def mark_verified(
    conn: sqlite3.Connection,
    memory_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark a memory as explicitly verified. Returns True on success."""
    return _mark_status(conn, memory_id, "verified", profile=profile, now=now)


def mark_flagged(
    conn: sqlite3.Connection,
    memory_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark a memory as flagged (suspicious or needs review)."""
    return _mark_status(conn, memory_id, "flagged", profile=profile, now=now)


def mark_contradicts(
    conn: sqlite3.Connection,
    memory_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark a memory as contradicted by another memory."""
    return _mark_status(
        conn, memory_id, "contradicts", profile=profile, now=now
    )


# ─── Confidence decay ───────────────────────────────────────────────────


def decay_memory_confidence(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_CONFIDENCE_FLOOR,
    initial: float = DEFAULT_INITIAL_CONFIDENCE,
) -> int:
    """Apply exponential confidence decay to eligible memories.

    For each memory with origin in ``extracted``/``inferred``:

      target = max(floor, initial * 2 ** (-elapsed/half_life))

    where ``elapsed`` is the time since ``accessed_at``. A row is
    updated only if the target is meaningfully lower than the
    current confidence (epsilon guard to prevent pointless writes).

    Returns the number of rows actually updated.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Build a placeholder-list of decayable origins for the IN clause
    placeholders = ",".join("?" * len(_DECAYABLE_ORIGINS))
    params = [profile] + list(_DECAYABLE_ORIGINS)

    rows = conn.execute(
        f"""
        SELECT id, confidence, accessed_at
        FROM memories
        WHERE profile = ?
          AND deleted_at IS NULL
          AND origin IN ({placeholders})
        """,
        params,
    ).fetchall()

    updated = 0
    for row in rows:
        accessed = _parse_accessed_at(row["accessed_at"])
        if accessed is None:
            continue

        elapsed = (now - accessed).total_seconds() / 86400.0
        target = _compute_target_confidence(
            elapsed,
            initial=initial,
            half_life_days=half_life_days,
            floor=floor,
        )

        current = float(row["confidence"])
        if target < current - _DECAY_WRITE_EPSILON:
            conn.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?",
                (target, row["id"]),
            )
            updated += 1

    return updated


# ─── Stale flagging ─────────────────────────────────────────────────────


def flag_stale_memories(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> int:
    """Flag memories that are still 'pending' after ``stale_days``.

    Only pending memories are eligible. Verified, flagged, and
    contradicts are skipped (their status reflects a decision that
    shouldn't be auto-overridden).

    Returns the number of rows updated.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    cutoff = (now - timedelta(days=stale_days)).isoformat()

    cur = conn.execute(
        """
        UPDATE memories
        SET verification = 'flagged'
        WHERE profile = ?
          AND deleted_at IS NULL
          AND verification = 'pending'
          AND accessed_at < ?
        """,
        (profile, cutoff),
    )
    return cur.rowcount


# ─── Orchestrator ───────────────────────────────────────────────────────


def run_verification(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_CONFIDENCE_FLOOR,
    initial: float = DEFAULT_INITIAL_CONFIDENCE,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> VerificationResult:
    """Full verification pass: decay + stale flagging.

    Idempotent: running twice in a row on the same state produces
    zeros on the second call because decay targets are absolute
    and stale flagging transitions pending → flagged only.
    """
    start = time.monotonic()

    decayed = decay_memory_confidence(
        conn,
        profile=profile,
        now=now,
        half_life_days=half_life_days,
        floor=floor,
        initial=initial,
    )
    flagged = flag_stale_memories(
        conn,
        profile=profile,
        now=now,
        stale_days=stale_days,
    )

    return VerificationResult(
        decayed_count=decayed,
        flagged_stale_count=flagged,
        duration_seconds=time.monotonic() - start,
    )
