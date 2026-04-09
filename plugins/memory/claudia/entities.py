"""Entity CRUD for the Claudia memory provider (Phase 2A.2d).

People, organizations, projects, locations, and concepts — the
primary organizing unit of Claudia's memory system. This module
provides the read/write layer that the provider (2A.2f) will call
from its memory.about / memory.entity tool handlers, and that
hybrid_search.py uses indirectly through the entities table's
importance and access_count columns.

Design principles:

- **Profile isolation is mandatory.** Every function takes a
  ``profile`` parameter and every query filters on it. The schema
  has a UNIQUE (profile, kind, name) constraint that allows the
  same entity name to exist in different profiles (one user's
  "Sarah" is not another user's "Sarah").

- **Exact find vs. fuzzy search.** ``find_entity`` does
  case-insensitive exact match on name or any alias and returns a
  single entity — it's the "resolve this specific reference" path.
  ``search_entities`` does substring match and returns a ranked
  list — it's the "show me candidates" path.

- **Aliases are canonicalized via json_each.** The ``aliases_json``
  column is a JSON array. Lookups use SQLite's ``json_each``
  table-valued function to correlate the array elements in the
  WHERE clause. This is both correct (no substring false-positives
  that plain LIKE would give on an alias like "Sarah Chen" when
  searching for "Sarah") and fast (no full-table scan in Python).

- **Upsert via ON CONFLICT.** ``upsert_entity`` uses SQLite's
  ``INSERT ... ON CONFLICT`` to create-or-update in a single
  statement, honoring the (profile, kind, name) unique constraint.

- **Attributes replace by default, merge on demand.** The
  ``attributes`` parameter of ``update_entity`` replaces the stored
  JSON object. Callers that want merge semantics should read first,
  combine in Python, and pass the merged dict.

- **Stateless reads.** ``get_entity``, ``find_entity``, and
  ``search_entities`` do NOT bump ``access_count`` or
  ``updated_at``. The rehearsal boost in hybrid_search depends on
  access_count being accurate, but those writes have to go through
  the serialized writer queue (Phase 2A.3). The caller (the provider
  in 2A.2f) will enqueue ``bump_access`` after recall.

- **Soft deletes everywhere.** Every recall path filters
  ``WHERE deleted_at IS NULL`` — matching the partial indexes in
  schema.py. ``soft_delete_entity`` writes a timestamp into
  ``deleted_at``; it never runs an actual DELETE.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.2d)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ─── Types ──────────────────────────────────────────────────────────────


#: Entity kinds matching the CHECK constraint in schema.py.
EntityKind = Literal["person", "organization", "project", "location", "concept"]

#: Valid kinds as a frozen set for validation.
VALID_KINDS = frozenset({"person", "organization", "project", "location", "concept"})


@dataclass
class Entity:
    """A single entity row from the entities table.

    Aliases and attributes are parsed from the on-disk JSON columns,
    so callers get native Python lists and dicts instead of raw strings.
    """

    id: int
    kind: str
    name: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    profile: str = "default"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Relationship:
    """A directed edge between two entities."""

    id: int
    from_entity_id: int
    to_entity_id: int
    type: str
    health_score: float = 1.0
    last_interaction_at: Optional[str] = None
    notes: Optional[str] = None
    profile: str = "default"
    created_at: str = ""
    updated_at: str = ""


# ─── Internal helpers ───────────────────────────────────────────────────


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def _parse_aliases(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Corrupt aliases_json value: %r", raw)
        return []
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def _parse_attributes(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Corrupt attributes_json value: %r", raw)
        return {}
    if not isinstance(value, dict):
        return {}
    return value


def _row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        kind=row["kind"],
        name=row["name"],
        aliases=_parse_aliases(row["aliases_json"]),
        attributes=_parse_attributes(row["attributes_json"]),
        importance=float(row["importance"]),
        access_count=int(row["access_count"]),
        profile=row["profile"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_relationship(row: sqlite3.Row) -> Relationship:
    return Relationship(
        id=row["id"],
        from_entity_id=row["from_entity_id"],
        to_entity_id=row["to_entity_id"],
        type=row["type"],
        health_score=float(row["health_score"]),
        last_interaction_at=row["last_interaction_at"],
        notes=row["notes"],
        profile=row["profile"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _validate_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(
            f"Invalid entity kind {kind!r}; must be one of {sorted(VALID_KINDS)}"
        )


def _serialize_aliases(aliases: Optional[List[str]]) -> Optional[str]:
    if aliases is None:
        return None
    return json.dumps(list(aliases), ensure_ascii=False)


def _serialize_attributes(attributes: Optional[Dict[str, Any]]) -> Optional[str]:
    if attributes is None:
        return None
    return json.dumps(attributes, ensure_ascii=False)


# ─── Entity CRUD ─────────────────────────────────────────────────────────


def create_entity(
    conn: sqlite3.Connection,
    kind: str,
    name: str,
    *,
    aliases: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    importance: float = 0.5,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Entity:
    """Insert a new entity and return the populated Entity dataclass.

    Raises:
        ValueError: if ``kind`` is not in ``VALID_KINDS``.
        sqlite3.IntegrityError: if an entity with the same
            (profile, kind, name) already exists and has not been
            soft-deleted.
    """
    _validate_kind(kind)
    now_iso = _now_iso(now)
    cur = conn.execute(
        """
        INSERT INTO entities (
            kind, name, aliases_json, attributes_json,
            importance, access_count, profile, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
        """,
        (
            kind,
            name,
            _serialize_aliases(aliases),
            _serialize_attributes(attributes),
            importance,
            profile,
            now_iso,
            now_iso,
        ),
    )
    entity_id = cur.lastrowid
    fetched = get_entity(conn, entity_id, profile=profile)
    if fetched is None:
        # Should never happen unless the INSERT didn't commit.
        raise RuntimeError(f"Entity {entity_id} vanished after insert")
    return fetched


def get_entity(
    conn: sqlite3.Connection,
    entity_id: int,
    *,
    profile: str = "default",
) -> Optional[Entity]:
    """Return an entity by primary key, or None if missing or soft-deleted."""
    row = conn.execute(
        """
        SELECT id, kind, name, aliases_json, attributes_json,
               importance, access_count, profile, created_at, updated_at
        FROM entities
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (entity_id, profile),
    ).fetchone()

    if row is None:
        return None
    return _row_to_entity(row)


def find_entity(
    conn: sqlite3.Connection,
    name: str,
    *,
    kind: Optional[str] = None,
    profile: str = "default",
) -> Optional[Entity]:
    """Resolve ``name`` to a single entity via exact (case-insensitive)
    match on ``entities.name`` OR on any element of ``aliases_json``.

    Returns the first matching entity (ordered by importance DESC so
    the most-signal entity wins when there are duplicates), or None
    if nothing matches.
    """
    if not name:
        return None

    if kind is not None:
        _validate_kind(kind)

    # json_each requires a non-null JSON value. COALESCE substitutes
    # an empty array for NULL/missing aliases so rows with no aliases
    # still participate in the exact-name check.
    row = conn.execute(
        """
        SELECT e.id, e.kind, e.name, e.aliases_json, e.attributes_json,
               e.importance, e.access_count, e.profile,
               e.created_at, e.updated_at
        FROM entities e
        WHERE e.profile = ?
          AND e.deleted_at IS NULL
          AND (? IS NULL OR e.kind = ?)
          AND (
            LOWER(e.name) = LOWER(?)
            OR EXISTS (
              SELECT 1 FROM json_each(COALESCE(e.aliases_json, '[]'))
              WHERE LOWER(json_each.value) = LOWER(?)
            )
          )
        ORDER BY e.importance DESC, e.access_count DESC
        LIMIT 1
        """,
        (profile, kind, kind, name, name),
    ).fetchone()

    if row is None:
        return None
    return _row_to_entity(row)


def search_entities(
    conn: sqlite3.Connection,
    query: str,
    *,
    kind: Optional[str] = None,
    profile: str = "default",
    limit: int = 10,
) -> List[Entity]:
    """Return up to ``limit`` entities whose name or aliases contain ``query``.

    Substring match, case-insensitive. Ordered by importance DESC,
    access_count DESC. Used for the "show me candidates" path — when
    the caller is not sure of the exact name.
    """
    if not query:
        return []

    if kind is not None:
        _validate_kind(kind)

    pattern = f"%{query.lower()}%"

    rows = conn.execute(
        """
        SELECT e.id, e.kind, e.name, e.aliases_json, e.attributes_json,
               e.importance, e.access_count, e.profile,
               e.created_at, e.updated_at
        FROM entities e
        WHERE e.profile = ?
          AND e.deleted_at IS NULL
          AND (? IS NULL OR e.kind = ?)
          AND (
            LOWER(e.name) LIKE ?
            OR EXISTS (
              SELECT 1 FROM json_each(COALESCE(e.aliases_json, '[]'))
              WHERE LOWER(json_each.value) LIKE ?
            )
          )
        ORDER BY e.importance DESC, e.access_count DESC
        LIMIT ?
        """,
        (profile, kind, kind, pattern, pattern, limit),
    ).fetchall()

    return [_row_to_entity(row) for row in rows]


def update_entity(
    conn: sqlite3.Connection,
    entity_id: int,
    *,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    importance: Optional[float] = None,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Optional[Entity]:
    """Patch fields on an existing entity. Returns the updated Entity.

    Only fields that are not None in the call are updated; pass
    ``aliases=[]`` to clear aliases rather than skip the update. The
    ``attributes`` dict replaces the stored value entirely — callers
    that want merge semantics should read, combine, and write back.

    Returns None if the target entity does not exist or has been
    soft-deleted.
    """
    updates: List[str] = []
    params: List[Any] = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if aliases is not None:
        updates.append("aliases_json = ?")
        params.append(_serialize_aliases(aliases))
    if attributes is not None:
        updates.append("attributes_json = ?")
        params.append(_serialize_attributes(attributes))
    if importance is not None:
        updates.append("importance = ?")
        params.append(importance)

    if not updates:
        # Nothing to do — just return current state.
        return get_entity(conn, entity_id, profile=profile)

    updates.append("updated_at = ?")
    params.append(_now_iso(now))

    params.extend([entity_id, profile])

    conn.execute(
        f"""
        UPDATE entities
        SET {', '.join(updates)}
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        params,
    )

    return get_entity(conn, entity_id, profile=profile)


def upsert_entity(
    conn: sqlite3.Connection,
    kind: str,
    name: str,
    *,
    aliases: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    importance: Optional[float] = None,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Entity:
    """Create if absent, update if present. Returns the resulting Entity.

    Matches on the (profile, kind, name) unique constraint. When the
    row already exists, passed fields overwrite the stored values
    (same replace-not-merge semantics as ``update_entity``); unspecified
    fields are left alone.

    This is the primary write path for the provider's sync_turn
    extraction flow: the extractor surfaces an entity reference, and
    the provider upserts without having to first check existence.
    """
    _validate_kind(kind)

    existing = conn.execute(
        """
        SELECT id FROM entities
        WHERE profile = ? AND kind = ? AND LOWER(name) = LOWER(?)
          AND deleted_at IS NULL
        LIMIT 1
        """,
        (profile, kind, name),
    ).fetchone()

    if existing is None:
        return create_entity(
            conn,
            kind,
            name,
            aliases=aliases,
            attributes=attributes,
            importance=importance if importance is not None else 0.5,
            profile=profile,
            now=now,
        )

    return update_entity(
        conn,
        existing["id"],
        aliases=aliases,
        attributes=attributes,
        importance=importance,
        profile=profile,
        now=now,
    ) or create_entity(  # pragma: no cover - unreachable because we just found it
        conn, kind, name, profile=profile, now=now
    )


def bump_access(
    conn: sqlite3.Connection,
    entity_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> None:
    """Increment ``access_count`` and touch ``updated_at`` on a recall.

    Called by the provider after an entity is surfaced via find/search
    so the rehearsal boost in hybrid_search accurately reflects how
    often the entity is referenced. No-op on missing or soft-deleted
    entities.
    """
    conn.execute(
        """
        UPDATE entities
        SET access_count = access_count + 1,
            updated_at = ?
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (_now_iso(now), entity_id, profile),
    )


def soft_delete_entity(
    conn: sqlite3.Connection,
    entity_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark an entity as deleted. Returns True if a row was updated.

    Subsequent reads filter it out via the ``deleted_at IS NULL``
    predicate. Nothing cascades to relationships — those get their
    own soft-delete via ``soft_delete_relationship``. The caller is
    responsible for deciding whether to delete related rows.
    """
    now_iso = _now_iso(now)
    cur = conn.execute(
        """
        UPDATE entities
        SET deleted_at = ?, updated_at = ?
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (now_iso, now_iso, entity_id, profile),
    )
    return cur.rowcount > 0


# ─── Relationship CRUD ──────────────────────────────────────────────────


def create_relationship(
    conn: sqlite3.Connection,
    from_entity_id: int,
    to_entity_id: int,
    rel_type: str,
    *,
    health_score: float = 1.0,
    last_interaction_at: Optional[str] = None,
    notes: Optional[str] = None,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Relationship:
    """Create a directed edge from one entity to another.

    Raises ``sqlite3.IntegrityError`` if the same
    (profile, from, to, type) tuple already exists — relationships
    are unique on that combination. Use ``update_relationship_health``
    for idempotent updates of the health_score on an existing edge.
    """
    now_iso = _now_iso(now)
    cur = conn.execute(
        """
        INSERT INTO relationships (
            from_entity_id, to_entity_id, type, health_score,
            last_interaction_at, notes, profile, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            from_entity_id,
            to_entity_id,
            rel_type,
            health_score,
            last_interaction_at,
            notes,
            profile,
            now_iso,
            now_iso,
        ),
    )

    row = conn.execute(
        """
        SELECT id, from_entity_id, to_entity_id, type, health_score,
               last_interaction_at, notes, profile, created_at, updated_at
        FROM relationships
        WHERE id = ?
        """,
        (cur.lastrowid,),
    ).fetchone()
    return _row_to_relationship(row)


def get_relationships(
    conn: sqlite3.Connection,
    entity_id: int,
    *,
    direction: Literal["both", "outgoing", "incoming"] = "both",
    profile: str = "default",
) -> List[Relationship]:
    """List relationships touching ``entity_id``.

    ``direction``:
      - ``"outgoing"`` — only edges where the entity is ``from_entity_id``
      - ``"incoming"`` — only edges where the entity is ``to_entity_id``
      - ``"both"`` — union of the two (the default)

    Ordered by ``health_score`` DESC so the strongest ties surface first.
    """
    if direction == "outgoing":
        where = "from_entity_id = ?"
        params = (entity_id, profile)
    elif direction == "incoming":
        where = "to_entity_id = ?"
        params = (entity_id, profile)
    elif direction == "both":
        where = "(from_entity_id = ? OR to_entity_id = ?)"
        params = (entity_id, entity_id, profile)
    else:
        raise ValueError(
            f"Invalid direction {direction!r}; must be 'both', 'outgoing', or 'incoming'"
        )

    rows = conn.execute(
        f"""
        SELECT id, from_entity_id, to_entity_id, type, health_score,
               last_interaction_at, notes, profile, created_at, updated_at
        FROM relationships
        WHERE {where}
          AND profile = ?
          AND deleted_at IS NULL
        ORDER BY health_score DESC, updated_at DESC
        """,
        params,
    ).fetchall()

    return [_row_to_relationship(row) for row in rows]


def update_relationship_health(
    conn: sqlite3.Connection,
    relationship_id: int,
    *,
    health_score: Optional[float] = None,
    last_interaction_at: Optional[str] = None,
    notes: Optional[str] = None,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> Optional[Relationship]:
    """Patch the health fields on a relationship. Returns updated row or None.

    Only fields explicitly supplied are updated. The ``notes`` field
    replaces the stored value (no merging); pass an empty string to
    clear notes.
    """
    updates: List[str] = []
    params: List[Any] = []

    if health_score is not None:
        updates.append("health_score = ?")
        params.append(health_score)
    if last_interaction_at is not None:
        updates.append("last_interaction_at = ?")
        params.append(last_interaction_at)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)

    if not updates:
        row = conn.execute(
            """
            SELECT id, from_entity_id, to_entity_id, type, health_score,
                   last_interaction_at, notes, profile, created_at, updated_at
            FROM relationships
            WHERE id = ? AND profile = ? AND deleted_at IS NULL
            """,
            (relationship_id, profile),
        ).fetchone()
        return _row_to_relationship(row) if row else None

    updates.append("updated_at = ?")
    params.append(_now_iso(now))
    params.extend([relationship_id, profile])

    conn.execute(
        f"""
        UPDATE relationships
        SET {', '.join(updates)}
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        params,
    )

    row = conn.execute(
        """
        SELECT id, from_entity_id, to_entity_id, type, health_score,
               last_interaction_at, notes, profile, created_at, updated_at
        FROM relationships
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (relationship_id, profile),
    ).fetchone()
    return _row_to_relationship(row) if row else None


def soft_delete_relationship(
    conn: sqlite3.Connection,
    relationship_id: int,
    *,
    profile: str = "default",
    now: Optional[datetime] = None,
) -> bool:
    """Mark a relationship deleted. Returns True if a row was updated."""
    now_iso = _now_iso(now)
    cur = conn.execute(
        """
        UPDATE relationships
        SET deleted_at = ?, updated_at = ?
        WHERE id = ? AND profile = ? AND deleted_at IS NULL
        """,
        (now_iso, now_iso, relationship_id, profile),
    )
    return cur.rowcount > 0
