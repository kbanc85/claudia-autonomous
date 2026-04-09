"""SQLite schema for the Claudia memory provider (Phase 2A.2a).

Defines the DDL for every table Claudia's hybrid memory system needs,
plus a tiny migration runner that tracks applied schema versions and
enables WAL mode on database open.

Design principles:

- **WAL mode always on.** Required for concurrent readers + single writer
  per the Phase 2A.3 concurrency design. Set at connection time via
  ``PRAGMA journal_mode=WAL``.
- **No sqlite-vec dependency in this module.** The vector similarity
  columns are stored as raw blobs; the hybrid search module
  (``hybrid_search.py``, sub-task 2A.2c) handles vector math with or
  without the vec0 extension, per the three-tier offline degradation.
- **Importance and access_count on every recallable row.** Feeds the
  25% importance weight and the rehearsal boost in hybrid scoring.
- **Provenance is mandatory.** Every memory and commitment carries a
  ``source_type`` and ``source_ref`` pointing at where the fact
  originated. Supports the trust-north-star principle.
- **Soft deletes only.** ``deleted_at`` columns let us invalidate a row
  without losing audit history. The recall path MUST filter
  ``WHERE deleted_at IS NULL``.
- **Profile isolation via claudia_home.** Each profile gets its own
  SQLite file at ``CLAUDIA_HOME/memory/claudia.db``. This module does
  not care about the path — ``initialize()`` in ``provider.py`` owns
  directory resolution.

Migration strategy:

- One ``_meta`` row per schema version, tracking when it was applied.
- ``apply_schema(conn)`` is idempotent: it reads ``_meta`` and runs
  only the migrations that have not been applied yet.
- New migrations append to the ``MIGRATIONS`` list below. Never modify
  an existing migration; add a new one that ALTERs or MIGRATEs data.

Reference: docs/decisions/memory-provider-design.md
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


# ─── Migrations ─────────────────────────────────────────────────────────
#
# Each entry is (version_int, description, DDL_text).
# Applied in order. Version numbers MUST be strictly increasing.
# Never modify a released migration; add a new one instead.

_MIGRATION_1_CORE_TABLES = """
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('person', 'organization', 'project', 'location', 'concept')),
    name            TEXT NOT NULL,
    aliases_json    TEXT,                 -- JSON array of alternate names
    attributes_json TEXT,                 -- JSON object of arbitrary fields (title, email, etc.)
    importance      REAL NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    access_count    INTEGER NOT NULL DEFAULT 0,
    profile         TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT,                 -- soft delete
    UNIQUE(profile, kind, name)
);

CREATE INDEX IF NOT EXISTS idx_entities_profile_kind ON entities(profile, kind) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_entities_access_count ON entities(access_count DESC) WHERE deleted_at IS NULL;


CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY,
    content         TEXT NOT NULL,
    entity_id       INTEGER REFERENCES entities(id),
    origin          TEXT NOT NULL CHECK (origin IN ('user_stated', 'extracted', 'inferred', 'corrected')),
    confidence      REAL NOT NULL DEFAULT 0.7 CHECK (confidence BETWEEN 0 AND 1),
    importance      REAL NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    access_count    INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,                 -- vector bytes; NULL when embedding not available
    embedding_dim   INTEGER,              -- dimension of stored vector (for validation)
    source_type     TEXT,                 -- 'transcript', 'gmail', 'upload', 'capture', 'conversation'
    source_ref      TEXT,                 -- opaque ref (file path, session id, message id)
    corrected_from  INTEGER REFERENCES memories(id),  -- for origin='corrected' trails
    verification    TEXT NOT NULL DEFAULT 'pending' CHECK (verification IN ('pending', 'verified', 'flagged', 'contradicts')),
    profile         TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,
    accessed_at     TEXT NOT NULL,        -- last recall timestamp for recency scoring
    deleted_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_profile ON memories(profile) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at DESC) WHERE deleted_at IS NULL;


CREATE TABLE IF NOT EXISTS relationships (
    id              INTEGER PRIMARY KEY,
    from_entity_id  INTEGER NOT NULL REFERENCES entities(id),
    to_entity_id    INTEGER NOT NULL REFERENCES entities(id),
    type            TEXT NOT NULL,        -- 'colleague', 'client', 'investor', 'friend', 'reports-to', etc.
    health_score    REAL NOT NULL DEFAULT 1.0 CHECK (health_score BETWEEN 0 AND 1),
    last_interaction_at TEXT,             -- for health decay calculation
    notes           TEXT,
    profile         TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT,
    UNIQUE(profile, from_entity_id, to_entity_id, type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_entity_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_entity_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relationships_health ON relationships(health_score) WHERE deleted_at IS NULL;


CREATE TABLE IF NOT EXISTS commitments (
    id              INTEGER PRIMARY KEY,
    content         TEXT NOT NULL,        -- "send Sarah the proposal"
    owner_entity_id INTEGER REFERENCES entities(id),  -- who made the commitment (usually the user)
    target_entity_id INTEGER REFERENCES entities(id), -- who it's owed to
    deadline        TEXT,                 -- ISO 8601, NULL means open-ended
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'completed', 'overdue', 'dropped', 'superseded')),
    source_type     TEXT,                 -- provenance chain back to origin
    source_ref      TEXT,
    profile         TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT,
    deleted_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status, deadline) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_commitments_owner ON commitments(owner_entity_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_commitments_target ON commitments(target_entity_id) WHERE deleted_at IS NULL;


CREATE TABLE IF NOT EXISTS _meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);


-- FTS5 virtual table over memory content for the 15% FTS portion of hybrid ranking.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep the FTS index in sync with the memories table.
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


MIGRATIONS: List[Tuple[int, str, str]] = [
    (1, "initial core tables: entities, memories, relationships, commitments, _meta, memories_fts", _MIGRATION_1_CORE_TABLES),
]


# ─── Public API ─────────────────────────────────────────────────────────

def open_connection(
    db_path: Path,
    *,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection configured for Claudia memory.

    Sets WAL journal mode (required for the Phase 2A.3 concurrency
    design), enables foreign keys, and configures reasonable busy
    timeout.

    ``check_same_thread`` defaults to True (the Python sqlite3
    safe default), meaning the returned connection may only be
    used from the thread that created it. The reader pool in
    ``reader.py`` passes ``check_same_thread=False`` because its
    connections migrate between threads on every acquire/release
    cycle; the writer thread in ``writer.py`` uses the default
    because a single thread owns the connection for its lifetime.
    Everywhere else (tests, admin paths) uses the default.

    The caller is responsible for closing the connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        isolation_level=None,  # autocommit mode
        check_same_thread=check_same_thread,
    )
    conn.row_factory = sqlite3.Row

    # Pragma configuration. Run before any schema application.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")  # Safe with WAL; trades durability for throughput
    conn.execute("PRAGMA busy_timeout=5000")   # 5 second wait on write lock
    conn.execute("PRAGMA temp_store=MEMORY")

    return conn


def current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version, or 0 if none."""
    # _meta may not exist on a brand-new DB.
    try:
        row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row["value"]) if row else 0
    except sqlite3.OperationalError:
        # _meta table doesn't exist yet
        return 0


def apply_schema(conn: sqlite3.Connection) -> int:
    """Apply all pending migrations, return the new schema version.

    Idempotent: safe to call on every startup. Skips migrations that
    have already been applied based on the ``_meta`` table.
    """
    current = current_schema_version(conn)
    applied = 0

    for version, description, ddl in MIGRATIONS:
        if version <= current:
            continue

        logger.info("Applying Claudia memory schema migration %d: %s", version, description)

        try:
            conn.executescript(ddl)
        except sqlite3.Error as e:
            logger.error("Migration %d failed: %s", version, e)
            raise

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", str(version), now_iso),
        )
        applied += 1

    if applied > 0:
        logger.info("Claudia memory schema now at version %d (%d migrations applied)", MIGRATIONS[-1][0], applied)

    return MIGRATIONS[-1][0] if MIGRATIONS else 0


def ensure_database(db_path: Path) -> sqlite3.Connection:
    """Open the Claudia memory database and apply any pending migrations.

    Convenience wrapper around ``open_connection`` + ``apply_schema``.
    Returns the opened connection with schema at the latest version.

    Typical use from ``provider.py`` during ``initialize()``:

        from plugins.memory.claudia.schema import ensure_database
        db_path = Path(claudia_home) / "memory" / "claudia.db"
        self.conn = ensure_database(db_path)
    """
    conn = open_connection(db_path)
    apply_schema(conn)
    return conn


# ─── Utility / introspection ────────────────────────────────────────────

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if the named table exists (non-deleted)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def describe_schema(conn: sqlite3.Connection) -> dict:
    """Return a compact description of the current schema state.

    Used for diagnostics via ``claudia memory status`` (when Phase 2A.2f
    wires the provider into the CLI) and by tests.
    """
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if not row["name"].startswith("sqlite_")
    ]

    info: dict = {
        "schema_version": current_schema_version(conn),
        "tables": tables,
        "entities_count": 0,
        "memories_count": 0,
        "relationships_count": 0,
        "commitments_count": 0,
    }

    for table, key in (
        ("entities", "entities_count"),
        ("memories", "memories_count"),
        ("relationships", "relationships_count"),
        ("commitments", "commitments_count"),
    ):
        if table in tables:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE deleted_at IS NULL"
            ).fetchone()
            info[key] = int(row["n"]) if row else 0

    return info
