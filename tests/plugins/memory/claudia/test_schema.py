"""Unit tests for plugins/memory/claudia/schema.py (Phase 2A.4 porting).

Phase 2A.2a landed the schema module without tests (the commit note
said "tests deferred"). Phase 2A.4 ports the schema-relevant cases
from claudia/memory-daemon/tests/test_fts_hybrid.py so the DDL
and trigger behavior have regression coverage.

Covers:

- Migration runner idempotency (applies once, not again)
- Schema version recorded in _meta table after apply
- describe_schema introspection returns all expected tables
- FTS5 virtual table creation
- FTS5 auto-sync on INSERT via trigger
- FTS5 auto-sync on DELETE via trigger
- FTS5 auto-sync on UPDATE via trigger
- Porter stemming (run → running match)
- BM25 ranking returns negative numbers (more negative = better)
- WAL journal mode is enabled on open

Ported from:
  claudia/memory-daemon/tests/test_fts_hybrid.py
  (Commit f6c3de2, adapted to the Phase 2A.2a schema module)

Not ported (Phase 2B or later):
  - Consolidation dedup tests (test_alias_*, test_fuzzy_entity)
  - Entity merge, memory correction, invalidation
    (test_entity_lifecycle.py)
  - LLM consolidation, predictions, metrics (test_llm_*, test_metrics)
  - Vault PARA layout (test_vault_operations.py)
  - Briefing service, canvas generation, document filing
  - Scheduler, daemon lifecycle, health endpoint
  - vec0 extension loading (we use pure-Python cosine)
"""

from __future__ import annotations

import sqlite3

import pytest

from plugins.memory.claudia.schema import (
    MIGRATIONS,
    apply_schema,
    current_schema_version,
    describe_schema,
    ensure_database,
    open_connection,
    table_exists,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    """Fresh, migrated Claudia memory database."""
    conn = ensure_database(tmp_path / "schema.db")
    yield conn
    conn.close()


NOW_ISO = "2026-04-09T12:00:00+00:00"


def _insert_memory(conn, content: str, importance: float = 0.5) -> int:
    cur = conn.execute(
        """
        INSERT INTO memories (
            content, origin, confidence, importance, access_count,
            profile, created_at, accessed_at
        ) VALUES (?, 'user_stated', 0.9, ?, 0, 'default', ?, ?)
        """,
        (content, importance, NOW_ISO, NOW_ISO),
    )
    return int(cur.lastrowid)


# ─── Migration runner ───────────────────────────────────────────────────


class TestMigrationRunner:
    def test_fresh_db_at_latest_version(self, db):
        """ensure_database applies every migration on a brand-new file."""
        latest = MIGRATIONS[-1][0]
        assert current_schema_version(db) == latest

    def test_apply_schema_idempotent(self, db):
        """Re-applying on an up-to-date DB is a no-op."""
        before = current_schema_version(db)
        apply_schema(db)
        after = current_schema_version(db)
        assert before == after

    def test_schema_version_recorded_in_meta(self, db):
        row = db.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert int(row["value"]) == MIGRATIONS[-1][0]

    def test_current_version_on_empty_conn(self, tmp_path):
        """current_schema_version returns 0 on a brand-new unmigrated conn."""
        conn = open_connection(tmp_path / "empty.db")
        try:
            assert current_schema_version(conn) == 0
        finally:
            conn.close()


# ─── Schema introspection ──────────────────────────────────────────────


class TestDescribeSchema:
    def test_describes_all_core_tables(self, db):
        info = describe_schema(db)
        tables = set(info["tables"])
        assert "entities" in tables
        assert "memories" in tables
        assert "relationships" in tables
        assert "commitments" in tables
        assert "_meta" in tables
        assert "memories_fts" in tables

    def test_counts_zero_on_empty_db(self, db):
        info = describe_schema(db)
        assert info["entities_count"] == 0
        assert info["memories_count"] == 0
        assert info["relationships_count"] == 0
        assert info["commitments_count"] == 0

    def test_counts_reflect_inserts(self, db):
        _insert_memory(db, "one")
        _insert_memory(db, "two")
        info = describe_schema(db)
        assert info["memories_count"] == 2

    def test_table_exists(self, db):
        assert table_exists(db, "memories") is True
        assert table_exists(db, "definitely_not_a_table") is False


# ─── WAL mode ───────────────────────────────────────────────────────────


class TestWalMode:
    def test_wal_journal_mode_enabled(self, db):
        row = db.execute("PRAGMA journal_mode").fetchone()
        # PRAGMA journal_mode returns the current mode as the first column.
        assert str(row[0]).lower() == "wal"

    def test_foreign_keys_enabled(self, db):
        row = db.execute("PRAGMA foreign_keys").fetchone()
        assert int(row[0]) == 1


# ─── FTS5 trigger sync (ported from v1 test_fts_hybrid.py) ──────────────


class TestFtsTriggers:
    def test_insert_syncs_to_fts(self, db):
        """Ported from v1 test_fts5_auto_sync_insert."""
        _insert_memory(db, "Sarah loves chocolate ice cream")

        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'chocolate'"
        ).fetchall()
        assert len(rows) == 1

    def test_delete_syncs_to_fts(self, db):
        """Ported from v1 test_fts5_auto_sync_delete."""
        mid = _insert_memory(db, "Temporary note about budgets")

        # Verify it's in FTS
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'budgets'"
        ).fetchall()
        assert len(rows) == 1

        # Delete from memories — trigger should cascade to FTS
        db.execute("DELETE FROM memories WHERE id = ?", (mid,))

        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'budgets'"
        ).fetchall()
        assert len(rows) == 0

    def test_update_syncs_to_fts(self, db):
        """Ported from v1 test_fts5_update_sync."""
        mid = _insert_memory(db, "Original content about pandas")

        # Original term is searchable
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'pandas'"
        ).fetchall()
        assert len(rows) == 1

        # Update the content
        db.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            ("Updated content about koalas", mid),
        )

        # Old term no longer matches
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'pandas'"
        ).fetchall()
        assert len(rows) == 0

        # New term does match
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'koalas'"
        ).fetchall()
        assert len(rows) == 1


# ─── FTS5 behavior (ported from v1 test_fts_hybrid.py) ──────────────────


class TestFtsBehavior:
    def test_porter_stemming_matches_variants(self, db):
        """Porter stemmer: 'run' matches 'running'.

        Ported from v1 test_fts5_stemming. The schema declares
        ``tokenize='porter unicode61'`` so English variants collapse
        to the same stem at index time.
        """
        _insert_memory(db, "She was running late to the meeting")

        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'run'"
        ).fetchall()
        assert len(rows) == 1

    def test_bm25_ranking_returns_negative(self, db):
        """bm25() returns negative numbers where smaller = better.

        Ported from v1 test_fts5_multiple_results_ranking. This is
        the raw SQL-level behavior that hybrid_search.py relies on
        for its score normalization.
        """
        _insert_memory(db, "The project deadline is next Friday")
        _insert_memory(
            db, "Discussed the project timeline and upcoming deadline with the team"
        )
        _insert_memory(db, "Had coffee this morning")

        rows = db.execute(
            """
            SELECT m.id, bm25(memories_fts) AS rank
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.rowid
            WHERE memories_fts MATCH 'project deadline'
            ORDER BY bm25(memories_fts)
            """
        ).fetchall()

        # Two matches (not the coffee row)
        assert len(rows) == 2
        # Every rank is a negative float; the first (best) is <= the second
        assert all(row["rank"] < 0 for row in rows)
        assert rows[0]["rank"] <= rows[1]["rank"]

    def test_fts_respects_multiword_and_operator(self, db):
        """Bare multi-word FTS5 queries are AND by default.

        This is the reason hybrid_search.build_fts_query inserts
        explicit OR operators — FTS5's default AND is too strict for
        recall-oriented search.
        """
        _insert_memory(db, "alpha only content")
        _insert_memory(db, "beta only content")
        _insert_memory(db, "alpha and beta together")

        # Default AND: only rows containing BOTH terms match
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'alpha beta'"
        ).fetchall()
        assert len(rows) == 1

        # Explicit OR: all three rows match
        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'alpha OR beta'"
        ).fetchall()
        assert len(rows) == 3

    def test_fts_query_with_no_matches_returns_empty(self, db):
        _insert_memory(db, "pineapple express")

        rows = db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'nonexistent'"
        ).fetchall()
        assert rows == []
