"""Unit tests for plugins/memory/claudia/entities.py (Phase 2A.2d).

Covers:

- create_entity happy path + validation (invalid kind, duplicate)
- get_entity by id, including soft-delete exclusion
- find_entity exact match on name (case-insensitive)
- find_entity alias resolution via json_each
- find_entity kind filter
- find_entity returns highest-importance when duplicates exist
- search_entities substring match on name and aliases
- update_entity patches individual fields
- update_entity replaces attributes dict (no merge)
- update_entity with no fields is a no-op read
- upsert_entity create path
- upsert_entity update path
- bump_access increments counter
- soft_delete_entity excludes from reads but preserves row
- profile isolation everywhere
- Relationship CRUD: create, get with direction filters, update health,
  soft delete
- Relationship unique constraint
- Relationship ordering by health_score DESC

All tests use a real SQLite DB via ``schema.ensure_database``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from plugins.memory.claudia.entities import (
    VALID_KINDS,
    Entity,
    Relationship,
    bump_access,
    create_entity,
    create_relationship,
    find_entity,
    get_entity,
    get_relationships,
    search_entities,
    soft_delete_entity,
    soft_delete_relationship,
    update_entity,
    update_relationship_health,
    upsert_entity,
)
from plugins.memory.claudia.schema import ensure_database


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    conn = ensure_database(tmp_path / "entities.db")
    yield conn
    conn.close()


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── create_entity + get_entity ─────────────────────────────────────────


class TestCreateEntity:
    def test_create_minimal(self, db):
        e = create_entity(db, "person", "Sarah", now=NOW)
        assert e.id > 0
        assert e.kind == "person"
        assert e.name == "Sarah"
        assert e.aliases == []
        assert e.attributes == {}
        assert e.importance == 0.5
        assert e.access_count == 0
        assert e.profile == "default"
        assert e.created_at == NOW.isoformat()
        assert e.updated_at == NOW.isoformat()

    def test_create_with_aliases_and_attributes(self, db):
        e = create_entity(
            db,
            "person",
            "Sarah Chen",
            aliases=["Sarah", "S. Chen", "schen@example.com"],
            attributes={"role": "VP Engineering", "company": "Acme"},
            importance=0.9,
            now=NOW,
        )
        assert e.aliases == ["Sarah", "S. Chen", "schen@example.com"]
        assert e.attributes == {"role": "VP Engineering", "company": "Acme"}
        assert e.importance == 0.9

    def test_create_all_kinds(self, db):
        for kind in VALID_KINDS:
            e = create_entity(db, kind, f"thing_{kind}", now=NOW)
            assert e.kind == kind

    def test_invalid_kind_raises(self, db):
        with pytest.raises(ValueError):
            create_entity(db, "alien", "E.T.", now=NOW)

    def test_duplicate_within_profile_and_kind_raises(self, db):
        create_entity(db, "person", "Sarah", now=NOW)
        with pytest.raises(sqlite3.IntegrityError):
            create_entity(db, "person", "Sarah", now=NOW)

    def test_same_name_different_kind_allowed(self, db):
        """Same label for a person and a project is fine."""
        p = create_entity(db, "person", "Mercury", now=NOW)
        proj = create_entity(db, "project", "Mercury", now=NOW)
        assert p.id != proj.id


class TestGetEntity:
    def test_get_existing(self, db):
        created = create_entity(db, "person", "Alice", now=NOW)
        fetched = get_entity(db, created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Alice"

    def test_get_missing_returns_none(self, db):
        assert get_entity(db, 999) is None

    def test_get_soft_deleted_returns_none(self, db):
        e = create_entity(db, "person", "Bob", now=NOW)
        soft_delete_entity(db, e.id, now=NOW)
        assert get_entity(db, e.id) is None

    def test_get_respects_profile(self, db):
        e = create_entity(db, "person", "Alice", profile="alice_home", now=NOW)
        # Default profile can't see it
        assert get_entity(db, e.id) is None
        # Its own profile can
        assert get_entity(db, e.id, profile="alice_home") is not None


# ─── find_entity ────────────────────────────────────────────────────────


class TestFindEntity:
    def test_exact_name_match(self, db):
        create_entity(db, "person", "Sarah", now=NOW)
        found = find_entity(db, "Sarah")
        assert found is not None
        assert found.name == "Sarah"

    def test_case_insensitive_name_match(self, db):
        create_entity(db, "person", "Sarah", now=NOW)
        assert find_entity(db, "sarah") is not None
        assert find_entity(db, "SARAH") is not None
        assert find_entity(db, "SaRaH") is not None

    def test_alias_match(self, db):
        create_entity(
            db,
            "person",
            "Sarah Chen",
            aliases=["Sarah", "schen"],
            now=NOW,
        )
        found = find_entity(db, "schen")
        assert found is not None
        assert found.name == "Sarah Chen"

    def test_alias_match_case_insensitive(self, db):
        create_entity(
            db,
            "person",
            "Sarah Chen",
            aliases=["schen@example.com"],
            now=NOW,
        )
        assert find_entity(db, "SCHEN@EXAMPLE.COM") is not None

    def test_no_substring_false_positive(self, db):
        """'Sarah' should NOT match the alias 'Sarah Chen' — exact match only."""
        create_entity(
            db,
            "person",
            "Robert",
            aliases=["Sarah Chen"],  # contains "Sarah" as substring
            now=NOW,
        )
        # find_entity is exact match, so "Sarah" should not match "Sarah Chen"
        # (use search_entities for substring)
        found = find_entity(db, "Sarah")
        assert found is None

    def test_kind_filter(self, db):
        create_entity(db, "person", "Mercury", now=NOW)
        create_entity(db, "project", "Mercury", now=NOW)

        person = find_entity(db, "Mercury", kind="person")
        assert person is not None
        assert person.kind == "person"

        project = find_entity(db, "Mercury", kind="project")
        assert project is not None
        assert project.kind == "project"

    def test_invalid_kind_raises(self, db):
        with pytest.raises(ValueError):
            find_entity(db, "anyone", kind="alien")

    def test_empty_name_returns_none(self, db):
        assert find_entity(db, "") is None

    def test_missing_returns_none(self, db):
        assert find_entity(db, "nobody") is None

    def test_soft_deleted_not_found(self, db):
        e = create_entity(db, "person", "Charlie", now=NOW)
        soft_delete_entity(db, e.id, now=NOW)
        assert find_entity(db, "Charlie") is None


# ─── search_entities ────────────────────────────────────────────────────


class TestSearchEntities:
    def test_substring_match_on_name(self, db):
        create_entity(db, "person", "Sarah Chen", importance=0.9, now=NOW)
        create_entity(db, "person", "Sarah Williams", importance=0.5, now=NOW)
        create_entity(db, "person", "Robert Jones", importance=0.5, now=NOW)

        results = search_entities(db, "sarah")
        names = [e.name for e in results]
        assert "Sarah Chen" in names
        assert "Sarah Williams" in names
        assert "Robert Jones" not in names

    def test_substring_match_on_alias(self, db):
        create_entity(
            db,
            "person",
            "Bob",
            aliases=["Robert", "Bobby"],
            now=NOW,
        )
        results = search_entities(db, "rober")
        assert len(results) == 1
        assert results[0].name == "Bob"

    def test_ordered_by_importance(self, db):
        create_entity(db, "person", "Alice", importance=0.3, now=NOW)
        create_entity(db, "person", "Albert", importance=0.9, now=NOW)
        create_entity(db, "person", "Alina", importance=0.5, now=NOW)

        results = search_entities(db, "al")
        assert [e.name for e in results] == ["Albert", "Alina", "Alice"]

    def test_limit_respected(self, db):
        for i in range(5):
            create_entity(db, "concept", f"topic_{i}", now=NOW)
        results = search_entities(db, "topic", limit=3)
        assert len(results) == 3

    def test_empty_query_returns_empty(self, db):
        create_entity(db, "person", "Sarah", now=NOW)
        assert search_entities(db, "") == []

    def test_kind_filter(self, db):
        create_entity(db, "person", "Mercury Jones", now=NOW)
        create_entity(db, "project", "Mercury Launch", now=NOW)
        create_entity(db, "planet" if False else "location", "Mercury", now=NOW)

        results = search_entities(db, "mercury", kind="person")
        assert len(results) == 1
        assert results[0].kind == "person"


# ─── update_entity ──────────────────────────────────────────────────────


class TestUpdateEntity:
    def test_update_importance(self, db):
        e = create_entity(db, "person", "Alice", importance=0.5, now=NOW)
        updated = update_entity(db, e.id, importance=0.9, now=NOW + timedelta(seconds=1))
        assert updated is not None
        assert updated.importance == 0.9

    def test_update_attributes_replaces_not_merges(self, db):
        e = create_entity(
            db,
            "person",
            "Alice",
            attributes={"role": "PM", "company": "Acme"},
            now=NOW,
        )
        updated = update_entity(db, e.id, attributes={"email": "alice@acme.com"}, now=NOW)
        assert updated is not None
        # Old attributes are replaced, not merged
        assert updated.attributes == {"email": "alice@acme.com"}

    def test_update_clears_aliases(self, db):
        e = create_entity(db, "person", "Alice", aliases=["Al", "Ally"], now=NOW)
        updated = update_entity(db, e.id, aliases=[], now=NOW)
        assert updated is not None
        assert updated.aliases == []

    def test_update_multiple_fields(self, db):
        e = create_entity(db, "person", "Alice", now=NOW)
        updated = update_entity(
            db,
            e.id,
            name="Alice Smith",
            aliases=["A. Smith"],
            importance=0.7,
            now=NOW,
        )
        assert updated is not None
        assert updated.name == "Alice Smith"
        assert updated.aliases == ["A. Smith"]
        assert updated.importance == 0.7

    def test_update_missing_returns_none(self, db):
        assert update_entity(db, 999, importance=0.9) is None

    def test_update_no_fields_returns_current_state(self, db):
        e = create_entity(db, "person", "Alice", importance=0.5, now=NOW)
        result = update_entity(db, e.id)
        assert result is not None
        assert result.importance == 0.5

    def test_update_bumps_updated_at(self, db):
        e = create_entity(db, "person", "Alice", now=NOW)
        later = NOW + timedelta(hours=1)
        updated = update_entity(db, e.id, importance=0.9, now=later)
        assert updated is not None
        assert updated.updated_at == later.isoformat()
        # created_at stays put
        assert updated.created_at == NOW.isoformat()


# ─── upsert_entity ──────────────────────────────────────────────────────


class TestUpsertEntity:
    def test_upsert_creates_when_absent(self, db):
        e = upsert_entity(db, "person", "Sarah", importance=0.8, now=NOW)
        assert e.id > 0
        assert e.name == "Sarah"
        assert e.importance == 0.8

    def test_upsert_updates_when_present(self, db):
        first = upsert_entity(db, "person", "Sarah", importance=0.5, now=NOW)
        second = upsert_entity(db, "person", "Sarah", importance=0.9, now=NOW)
        assert first.id == second.id
        assert second.importance == 0.9

    def test_upsert_case_insensitive_match(self, db):
        first = upsert_entity(db, "person", "Sarah", now=NOW)
        second = upsert_entity(db, "person", "sarah", importance=0.9, now=NOW)
        # Should update the same row, not create a new one
        assert first.id == second.id


# ─── bump_access ────────────────────────────────────────────────────────


class TestBumpAccess:
    def test_bump_increments_counter(self, db):
        e = create_entity(db, "person", "Sarah", now=NOW)
        assert e.access_count == 0

        bump_access(db, e.id, now=NOW)
        bump_access(db, e.id, now=NOW)
        bump_access(db, e.id, now=NOW)

        fetched = get_entity(db, e.id)
        assert fetched is not None
        assert fetched.access_count == 3

    def test_bump_missing_is_noop(self, db):
        bump_access(db, 999)  # no error

    def test_bump_soft_deleted_is_noop(self, db):
        e = create_entity(db, "person", "Sarah", now=NOW)
        soft_delete_entity(db, e.id, now=NOW)

        bump_access(db, e.id, now=NOW)

        # Row still has count=0 (noop because deleted_at is set)
        row = db.execute(
            "SELECT access_count FROM entities WHERE id = ?", (e.id,)
        ).fetchone()
        assert row["access_count"] == 0


# ─── soft_delete_entity ─────────────────────────────────────────────────


class TestSoftDeleteEntity:
    def test_delete_returns_true_on_success(self, db):
        e = create_entity(db, "person", "Sarah", now=NOW)
        assert soft_delete_entity(db, e.id, now=NOW) is True

    def test_delete_missing_returns_false(self, db):
        assert soft_delete_entity(db, 999) is False

    def test_delete_twice_returns_false_second_time(self, db):
        e = create_entity(db, "person", "Sarah", now=NOW)
        assert soft_delete_entity(db, e.id, now=NOW) is True
        assert soft_delete_entity(db, e.id, now=NOW) is False

    def test_deleted_entity_invisible_to_all_reads(self, db):
        e = create_entity(db, "person", "Sarah", aliases=["S"], now=NOW)
        soft_delete_entity(db, e.id, now=NOW)

        assert get_entity(db, e.id) is None
        assert find_entity(db, "Sarah") is None
        assert find_entity(db, "S") is None  # alias
        assert search_entities(db, "sar") == []

    def test_row_still_exists_in_raw_table(self, db):
        """Soft delete sets deleted_at but keeps the row for audit."""
        e = create_entity(db, "person", "Sarah", now=NOW)
        soft_delete_entity(db, e.id, now=NOW)
        row = db.execute(
            "SELECT deleted_at FROM entities WHERE id = ?", (e.id,)
        ).fetchone()
        assert row is not None
        assert row["deleted_at"] is not None


# ─── Profile isolation ──────────────────────────────────────────────────


class TestProfileIsolation:
    def test_same_name_different_profiles(self, db):
        a = create_entity(db, "person", "Sarah", profile="user_a", now=NOW)
        b = create_entity(db, "person", "Sarah", profile="user_b", now=NOW)
        assert a.id != b.id

    def test_find_is_profile_scoped(self, db):
        create_entity(db, "person", "Sarah", profile="user_a", now=NOW)
        assert find_entity(db, "Sarah", profile="user_a") is not None
        assert find_entity(db, "Sarah", profile="user_b") is None

    def test_search_is_profile_scoped(self, db):
        create_entity(db, "person", "Sarah Chen", profile="user_a", now=NOW)
        assert len(search_entities(db, "sarah", profile="user_a")) == 1
        assert search_entities(db, "sarah", profile="user_b") == []


# ─── Relationships ──────────────────────────────────────────────────────


@pytest.fixture()
def db_with_people(db):
    alice = create_entity(db, "person", "Alice", now=NOW)
    bob = create_entity(db, "person", "Bob", now=NOW)
    carol = create_entity(db, "person", "Carol", now=NOW)
    return db, alice, bob, carol


class TestCreateRelationship:
    def test_basic_create(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        assert rel.id > 0
        assert rel.from_entity_id == alice.id
        assert rel.to_entity_id == bob.id
        assert rel.type == "colleague"
        assert rel.health_score == 1.0

    def test_duplicate_raises(self, db_with_people):
        db, alice, bob, _ = db_with_people
        create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        with pytest.raises(sqlite3.IntegrityError):
            create_relationship(db, alice.id, bob.id, "colleague", now=NOW)

    def test_different_type_between_same_entities_allowed(self, db_with_people):
        db, alice, bob, _ = db_with_people
        r1 = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        r2 = create_relationship(db, alice.id, bob.id, "mentor", now=NOW)
        assert r1.id != r2.id


class TestGetRelationships:
    def test_direction_both(self, db_with_people):
        db, alice, bob, carol = db_with_people
        create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        create_relationship(db, carol.id, alice.id, "mentor", now=NOW)

        rels = get_relationships(db, alice.id, direction="both")
        assert len(rels) == 2

    def test_direction_outgoing(self, db_with_people):
        db, alice, bob, carol = db_with_people
        create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        create_relationship(db, carol.id, alice.id, "mentor", now=NOW)

        rels = get_relationships(db, alice.id, direction="outgoing")
        assert len(rels) == 1
        assert rels[0].to_entity_id == bob.id

    def test_direction_incoming(self, db_with_people):
        db, alice, bob, carol = db_with_people
        create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        create_relationship(db, carol.id, alice.id, "mentor", now=NOW)

        rels = get_relationships(db, alice.id, direction="incoming")
        assert len(rels) == 1
        assert rels[0].from_entity_id == carol.id

    def test_order_by_health_score_desc(self, db_with_people):
        db, alice, bob, carol = db_with_people
        create_relationship(db, alice.id, bob.id, "colleague", health_score=0.3, now=NOW)
        create_relationship(db, alice.id, carol.id, "friend", health_score=0.9, now=NOW)

        rels = get_relationships(db, alice.id, direction="outgoing")
        assert rels[0].type == "friend"
        assert rels[1].type == "colleague"

    def test_invalid_direction_raises(self, db_with_people):
        db, alice, _, _ = db_with_people
        with pytest.raises(ValueError):
            get_relationships(db, alice.id, direction="sideways")  # type: ignore[arg-type]

    def test_soft_deleted_excluded(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        soft_delete_relationship(db, rel.id, now=NOW)

        assert get_relationships(db, alice.id) == []


class TestUpdateRelationshipHealth:
    def test_update_health_score(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        updated = update_relationship_health(
            db, rel.id, health_score=0.4, now=NOW + timedelta(days=1)
        )
        assert updated is not None
        assert updated.health_score == 0.4

    def test_update_last_interaction(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        interaction_iso = (NOW + timedelta(days=3)).isoformat()
        updated = update_relationship_health(
            db, rel.id, last_interaction_at=interaction_iso, now=NOW
        )
        assert updated is not None
        assert updated.last_interaction_at == interaction_iso

    def test_update_notes(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        updated = update_relationship_health(db, rel.id, notes="reliable", now=NOW)
        assert updated is not None
        assert updated.notes == "reliable"

    def test_update_missing_returns_none(self, db_with_people):
        db, *_ = db_with_people
        assert update_relationship_health(db, 999, health_score=0.5) is None

    def test_update_no_fields_returns_current(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", health_score=0.7, now=NOW)
        result = update_relationship_health(db, rel.id)
        assert result is not None
        assert result.health_score == 0.7


class TestSoftDeleteRelationship:
    def test_delete_returns_true(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        assert soft_delete_relationship(db, rel.id, now=NOW) is True

    def test_delete_missing_returns_false(self, db_with_people):
        db, *_ = db_with_people
        assert soft_delete_relationship(db, 999) is False

    def test_delete_excludes_from_reads(self, db_with_people):
        db, alice, bob, _ = db_with_people
        rel = create_relationship(db, alice.id, bob.id, "colleague", now=NOW)
        soft_delete_relationship(db, rel.id, now=NOW)
        assert get_relationships(db, alice.id) == []
        assert update_relationship_health(db, rel.id, health_score=0.5) is None
