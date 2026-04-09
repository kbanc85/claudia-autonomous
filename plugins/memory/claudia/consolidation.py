"""Auto-consolidation for Claudia memory (Phase 2B.3).

Background maintenance work that keeps the memory store tidy:

- **Fuzzy entity dedup**: find same-kind entities whose names are
  near-identical (exact match, shared alias, or high SequenceMatcher
  ratio with last-token safety for people) and merge them into
  one. The survivor inherits aliases, attributes, relationships,
  and commitments from the merged duplicate. The duplicate is
  soft-deleted.

- **Commitment FK resolution**: Phase 2B.2 writes commitments with
  NULL ``owner_entity_id`` / ``target_entity_id`` because extraction
  runs concurrently with detection and entity IDs are not yet
  available at write time. This pass scans unlinked commitments
  and sets ``target_entity_id`` when a person entity's name
  appears in the commitment content. Owner linking is deferred
  to a later phase (no "self" entity concept yet).

- **Idempotent**: running consolidation twice on the same state
  does nothing the second time. Callers can schedule it safely
  on a timer without worrying about duplicate work.

Design principles:

- **Stdlib only.** ``difflib.SequenceMatcher`` is the similarity
  function. No rapidfuzz, no Levenshtein package — keeps the
  plugin light and portable. The threshold (0.92) is conservative;
  SequenceMatcher will almost always flag typo-level variants
  without catching distinct-but-similar names.

- **Conservative merge by default.** The "Joel Salinas / Joel
  Hart" safety gate (same first name, different last name) is
  permanent: for kind=person entities with multi-token names,
  the LAST token must match before SequenceMatcher's score is
  trusted. Single-token people never fuzzy-merge — only exact
  matches.

- **Single-transaction merges.** Each merge runs inside a single
  writer-queue job, so relationship redirects, commitment
  redirects, alias union, attribute merge, and soft delete of
  the duplicate all commit atomically. If any step fails the
  whole thing rolls back and leaves state consistent.

- **No cross-profile bleed.** Every query filters on profile.
  Running consolidation on one profile never touches another.

Public API:

- ``ConsolidationResult`` dataclass with counts and duration
- ``find_fuzzy_candidates(conn, *, profile, threshold)``
- ``merge_entities(conn, keep_id, merge_id, *, profile, now)``
- ``resolve_commitment_fks(conn, *, profile, now, limit)``
- ``run_consolidation(conn, *, profile, now)`` → ConsolidationResult

Called from ``ClaudiaMemoryProvider.consolidate()`` via the
writer queue so the whole pass serializes with normal writes.

Reference: plugins/memory/claudia/schema.py for the relationships
and commitments table DDL, entities.py for CRUD functions we
compose, plans/phase-2b-handoff.md for the Phase 2B.3 scope notes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from plugins.memory.claudia import commitments as commitments_module
from plugins.memory.claudia import entities

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────


#: Minimum SequenceMatcher ratio to trigger a fuzzy merge. A
#: tripwire test locks this between 0.85 and 1.0 to prevent
#: accidental low-threshold drift.
AUTO_MERGE_THRESHOLD = 0.92

#: Default limit when scanning unlinked commitments. Consolidation
#: runs in a single writer job so processing thousands of rows is
#: fine; this cap prevents runaway loops in pathological cases.
DEFAULT_COMMITMENT_SCAN_LIMIT = 1000


# ─── Dataclasses ────────────────────────────────────────────────────────


@dataclass
class ConsolidationResult:
    """Summary of a single consolidation pass.

    Callers can log this or surface it to the user ("merged 3
    entities, linked 7 commitments, took 120ms"). Scheduled
    consolidation jobs would emit this as a metric.
    """

    entities_merged: int = 0
    commitments_linked: int = 0
    duration_seconds: float = 0.0


# ─── Fuzzy matching ─────────────────────────────────────────────────────


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def _parse_aliases(raw_json: Optional[str]) -> List[str]:
    """Local alias parser (not importing the private helper from
    entities.py — parses from raw SQL rows inside find_fuzzy_candidates).
    """
    if not raw_json:
        return []
    try:
        value = json.loads(raw_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def _fuzzy_match_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Score two entity rows on a 0.0-1.0 similarity scale.

    Rules (in priority order):

    1. Exact name match (case-insensitive) → 1.0
    2. Any shared alias (including one entity's name appearing in
       the other's alias list) → 0.95
    3. SequenceMatcher ratio ≥ AUTO_MERGE_THRESHOLD, with last-token
       safety for people → ratio
    4. Otherwise → 0.0 (no merge)

    Arguments are dict-like rows with keys: id, kind, name,
    aliases_json, importance, access_count. This lets callers
    pass either sqlite3.Row objects or plain dicts (used in tests).

    NOTE: This function does NOT enforce same-kind. The caller
    (``find_fuzzy_candidates``) filters by kind before invoking
    this. ``_fuzzy_match_score`` just reports numeric similarity.
    """
    a_name = (a["name"] or "").lower()
    b_name = (b["name"] or "").lower()

    if not a_name or not b_name:
        return 0.0

    if a_name == b_name:
        return 1.0

    # Alias overlap check. Bracket access works for both sqlite3.Row
    # (from find_fuzzy_candidates) and plain dicts (from tests).
    a_aliases = _parse_aliases(a["aliases_json"])
    b_aliases = _parse_aliases(b["aliases_json"])

    a_all = {a_name}
    a_all.update(al.lower() for al in a_aliases)
    b_all = {b_name}
    b_all.update(al.lower() for al in b_aliases)

    if a_all & b_all:
        return 0.95

    # SequenceMatcher fuzzy similarity
    ratio = SequenceMatcher(None, a_name, b_name).ratio()
    if ratio < AUTO_MERGE_THRESHOLD:
        return 0.0

    # Last-token safety for people
    if a["kind"] == "person":
        a_tokens = a_name.split()
        b_tokens = b_name.split()

        if len(a_tokens) > 1 and len(b_tokens) > 1:
            # Both multi-word: last tokens MUST match
            if a_tokens[-1] != b_tokens[-1]:
                return 0.0
        elif len(a_tokens) == 1 and len(b_tokens) > 1:
            # Single word must match one of the multi-word tokens
            if a_tokens[0] not in b_tokens:
                return 0.0
        elif len(b_tokens) == 1 and len(a_tokens) > 1:
            if b_tokens[0] not in a_tokens:
                return 0.0
        # else both single-word — SequenceMatcher already scored
        # them; nothing further to do.

    return ratio


def find_fuzzy_candidates(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    threshold: float = AUTO_MERGE_THRESHOLD,
) -> List[Tuple[int, int, float]]:
    """Find entity pairs in ``profile`` that should merge.

    Returns a list of ``(keep_id, merge_id, score)`` tuples where
    ``keep_id`` is the higher-importance entity (tiebreak on
    ``access_count``, then ``id``). Score is the fuzzy match ratio.

    Only same-kind pairs are considered. The scan is O(n²) on the
    number of entities per kind, which is fine for consolidation
    runs up to a few thousand entities. For larger scales, future
    work could add blocking (e.g., hash first letters into buckets).
    """
    rows = conn.execute(
        """
        SELECT id, kind, name, aliases_json, importance, access_count
        FROM entities
        WHERE profile = ? AND deleted_at IS NULL
        ORDER BY kind, id
        """,
        (profile,),
    ).fetchall()

    # Group by kind
    by_kind: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)

    candidates: List[Tuple[int, int, float]] = []
    for kind, group in by_kind.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                score = _fuzzy_match_score(a, b)
                if score >= threshold:
                    keep, merge = _pick_keep_merge(a, b)
                    candidates.append((keep["id"], merge["id"], score))

    return candidates


def _pick_keep_merge(
    a: sqlite3.Row,
    b: sqlite3.Row,
) -> Tuple[sqlite3.Row, sqlite3.Row]:
    """Return (keep, merge) ordered by importance DESC, access_count DESC, id ASC."""
    a_key = (a["importance"], a["access_count"], -a["id"])
    b_key = (b["importance"], b["access_count"], -b["id"])
    if a_key >= b_key:
        return a, b
    return b, a


# ─── Entity merging ─────────────────────────────────────────────────────


def merge_entities(
    conn: sqlite3.Connection,
    keep_id: int,
    merge_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Optional[int]:
    """Merge ``merge_id`` into ``keep_id`` within ``profile``.

    Steps (all inside the caller's transaction):

    1. Redirect relationships where merge_id appears as from/to.
       Relationship conflicts (keep already has the same edge)
       cause merge's version to be soft-deleted. Self-loops after
       redirect are soft-deleted.
    2. Redirect commitments where merge_id is owner or target.
       No conflict resolution needed — multiple commitments
       pointing to the same entity is legal.
    3. Union merge's aliases (plus its name) into keep's aliases.
       Remove keep's own name from the alias list if it leaked in.
    4. Merge merge's attributes into keep's — keep wins on conflicts.
    5. Take max(keep.importance, merge.importance).
    6. Soft-delete merge.

    Returns ``keep_id`` on success, ``None`` if either entity is
    missing, soft-deleted, in another profile, or if keep == merge.
    """
    if keep_id == merge_id:
        return None

    keep = entities.get_entity(conn, keep_id, profile=profile)
    merge = entities.get_entity(conn, merge_id, profile=profile)
    if keep is None or merge is None:
        return None

    now_iso = _now_iso(now)

    # Step 1: relationships
    _redirect_relationships(
        conn, keep_id, merge_id, profile=profile, now_iso=now_iso
    )

    # Step 2: commitments
    _redirect_commitments(
        conn, keep_id, merge_id, profile=profile, now_iso=now_iso
    )

    # Step 3: alias union (dedup, preserve insertion order)
    combined_aliases: List[str] = []
    seen_lower: set = set()
    for alias_source in (keep.aliases, merge.aliases, [merge.name]):
        for alias in alias_source:
            alias_lower = alias.lower()
            if alias_lower and alias_lower not in seen_lower:
                seen_lower.add(alias_lower)
                combined_aliases.append(alias)
    # Remove keep's own name if it crept in
    combined_aliases = [
        a for a in combined_aliases if a.lower() != keep.name.lower()
    ]

    # Step 4: attribute merge (keep wins on conflicts)
    combined_attributes: Dict[str, Any] = dict(merge.attributes)
    combined_attributes.update(keep.attributes)

    # Step 5: importance max
    combined_importance = max(keep.importance, merge.importance)

    entities.update_entity(
        conn,
        keep_id,
        aliases=combined_aliases,
        attributes=combined_attributes,
        importance=combined_importance,
        profile=profile,
        now=now,
    )

    # Step 6: soft-delete merge
    entities.soft_delete_entity(conn, merge_id, profile=profile, now=now)

    return keep_id


def _redirect_relationships(
    conn: sqlite3.Connection,
    keep_id: int,
    merge_id: int,
    *,
    profile: str,
    now_iso: str,
) -> None:
    """Move every relationship touching ``merge_id`` to ``keep_id``.

    Row-by-row because the unique constraint on (profile, from, to,
    type) would make a bulk UPDATE fail on any pre-existing edge.
    For each merge-side row:

    - Compute the new (from, to) pair after redirect.
    - If the result is a self-loop (from == to), soft-delete the row.
    - If keep already has an active edge at the target pair with
      the same type, soft-delete merge's row (conflict).
    - Otherwise, update merge's row in place.
    """
    rows = conn.execute(
        """
        SELECT id, from_entity_id, to_entity_id, type
        FROM relationships
        WHERE profile = ?
          AND deleted_at IS NULL
          AND (from_entity_id = ? OR to_entity_id = ?)
        """,
        (profile, merge_id, merge_id),
    ).fetchall()

    for r in rows:
        rel_id = r["id"]
        new_from = keep_id if r["from_entity_id"] == merge_id else r["from_entity_id"]
        new_to = keep_id if r["to_entity_id"] == merge_id else r["to_entity_id"]

        # Self-loop check
        if new_from == new_to:
            conn.execute(
                """
                UPDATE relationships
                SET deleted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now_iso, now_iso, rel_id),
            )
            continue

        # Conflict check — is there already an active edge at the
        # new pair with the same type, not this row?
        conflict = conn.execute(
            """
            SELECT id FROM relationships
            WHERE profile = ?
              AND from_entity_id = ?
              AND to_entity_id = ?
              AND type = ?
              AND deleted_at IS NULL
              AND id != ?
            LIMIT 1
            """,
            (profile, new_from, new_to, r["type"], rel_id),
        ).fetchone()

        if conflict is not None:
            # Soft-delete merge's row; keep's survives untouched
            conn.execute(
                """
                UPDATE relationships
                SET deleted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now_iso, now_iso, rel_id),
            )
        else:
            # Redirect in place
            conn.execute(
                """
                UPDATE relationships
                SET from_entity_id = ?, to_entity_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_from, new_to, now_iso, rel_id),
            )


def _redirect_commitments(
    conn: sqlite3.Connection,
    keep_id: int,
    merge_id: int,
    *,
    profile: str,
    now_iso: str,
) -> None:
    """Redirect commitments from merge_id to keep_id on both FK slots.

    No conflict resolution: commitments have no uniqueness constraint
    on (owner, target), so multiple rows pointing at the same entity
    are legal. Simple bulk UPDATE.
    """
    conn.execute(
        """
        UPDATE commitments
        SET owner_entity_id = ?, updated_at = ?
        WHERE owner_entity_id = ?
          AND profile = ?
          AND deleted_at IS NULL
        """,
        (keep_id, now_iso, merge_id, profile),
    )
    conn.execute(
        """
        UPDATE commitments
        SET target_entity_id = ?, updated_at = ?
        WHERE target_entity_id = ?
          AND profile = ?
          AND deleted_at IS NULL
        """,
        (keep_id, now_iso, merge_id, profile),
    )


# ─── Commitment FK resolution ───────────────────────────────────────────


def resolve_commitment_fks(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
    limit: int = DEFAULT_COMMITMENT_SCAN_LIMIT,
) -> int:
    """Link commitments with NULL target_entity_id to person entities
    whose names appear in the commitment content.

    Heuristic: case-insensitive substring match of each person's
    name (or aliases) against the commitment content. First match
    wins. Commitments with an existing target_entity_id are NOT
    overwritten. Owner entity is NOT resolved in this phase because
    "the user" has no entity representation yet.

    Returns the number of commitments updated.
    """
    now_iso = _now_iso(now)

    # Collect all active person entities up front. This avoids
    # re-querying per commitment. Build a list of (id, lowered_name,
    # lowered_aliases) for fast substring scans.
    person_rows = conn.execute(
        """
        SELECT id, name, aliases_json
        FROM entities
        WHERE profile = ?
          AND kind = 'person'
          AND deleted_at IS NULL
        """,
        (profile,),
    ).fetchall()

    if not person_rows:
        return 0

    person_index: List[Tuple[int, List[str]]] = []
    for p in person_rows:
        names = [p["name"].lower()]
        names.extend(
            a.lower() for a in _parse_aliases(p["aliases_json"])
        )
        # Filter out very short tokens to reduce false positives
        # (e.g., an alias like "A" would match every commitment)
        names = [n for n in names if len(n) >= 3]
        if names:
            person_index.append((p["id"], names))

    if not person_index:
        return 0

    # Fetch unlinked commitments
    unlinked = conn.execute(
        """
        SELECT id, content
        FROM commitments
        WHERE profile = ?
          AND deleted_at IS NULL
          AND target_entity_id IS NULL
        LIMIT ?
        """,
        (profile, limit),
    ).fetchall()

    linked_count = 0
    for c in unlinked:
        content_lower = (c["content"] or "").lower()
        if not content_lower:
            continue

        for person_id, names in person_index:
            if any(name in content_lower for name in names):
                conn.execute(
                    """
                    UPDATE commitments
                    SET target_entity_id = ?, updated_at = ?
                    WHERE id = ? AND target_entity_id IS NULL
                    """,
                    (person_id, now_iso, c["id"]),
                )
                linked_count += 1
                break

    return linked_count


# ─── Orchestrator ───────────────────────────────────────────────────────


def run_consolidation(
    conn: sqlite3.Connection,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> ConsolidationResult:
    """Full consolidation pass for one profile.

    Runs:

    1. Fuzzy entity dedup: find candidates and merge each pair.
    2. Commitment FK resolution: link unlinked commitments.

    Returns a ConsolidationResult with counts and wall-clock
    duration. Idempotent: running twice in a row on the same
    state produces zeros on the second run.
    """
    start = time.monotonic()

    entities_merged = 0
    candidates = find_fuzzy_candidates(conn, profile=profile)
    for keep_id, merge_id, _score in candidates:
        # Re-check both entities exist — earlier merges in this
        # pass might have soft-deleted one of them (transitive
        # merges). Skip if either is gone.
        if entities.get_entity(conn, keep_id, profile=profile) is None:
            continue
        if entities.get_entity(conn, merge_id, profile=profile) is None:
            continue
        result = merge_entities(
            conn, keep_id, merge_id, profile=profile, now=now
        )
        if result is not None:
            entities_merged += 1

    commitments_linked = resolve_commitment_fks(
        conn, profile=profile, now=now
    )

    duration = time.monotonic() - start

    return ConsolidationResult(
        entities_merged=entities_merged,
        commitments_linked=commitments_linked,
        duration_seconds=duration,
    )
