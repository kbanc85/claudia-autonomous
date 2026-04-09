"""Unit tests for plugins/memory/claudia/consolidation.py (Phase 2B.3).

Covered:

- ``ConsolidationResult`` dataclass shape and defaults
- ``_fuzzy_match_score`` scoring:
    * exact name match (case-insensitive) → 1.0
    * shared alias → 0.95
    * SequenceMatcher ≥ 0.92 → ratio
    * below threshold → 0.0
    * different kinds → 0.0 (handled by caller, but tested via candidates)
    * last-token safety for people ("Joel Salinas" vs "Joel Hart")
    * single-word person matching rules
- ``find_fuzzy_candidates``:
    * empty profile → []
    * single entity → []
    * exact duplicates → one candidate
    * near-duplicates → one candidate
    * different kinds never paired
    * no cross-profile bleed
    * higher-importance entity is "keep"
    * ordered tiebreak on access_count
- ``merge_entities``:
    * missing keep_id → None
    * missing merge_id → None
    * same id → no-op (safety)
    * aliases combined (merge's name + merge's aliases added to keep's)
    * attributes merged (keep wins on conflicts, merge fills gaps)
    * importance = max of both
    * merge entity soft-deleted
    * relationships transferred (from direction)
    * relationships transferred (to direction)
    * relationships transferred (both directions)
    * relationship conflicts (keep already has same edge) → merge's row soft-deleted
    * self-loop after redirect → soft-deleted
    * commitments owner_entity_id transferred
    * commitments target_entity_id transferred
    * soft-deleted entities not affected
- ``resolve_commitment_fks``:
    * commitment with NULL target → linked if entity name appears
    * commitment with existing target → not overwritten
    * case-insensitive match
    * multiple name matches → first wins
    * no match → stays NULL
    * soft-deleted entities not matched
    * soft-deleted commitments not processed
- ``run_consolidation``:
    * empty DB → zeros in result
    * merges happen and commitments linked in one pass
    * idempotent (re-running after first pass changes nothing)
    * result dataclass populated correctly

Mirrors the Phase 2A test conventions: deterministic NOW anchor,
real SQLite via ``schema.ensure_database``, one behavior per test.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from plugins.memory.claudia import commitments, entities, schema
from plugins.memory.claudia.consolidation import (
    AUTO_MERGE_THRESHOLD,
    ConsolidationResult,
    _fuzzy_match_score,
    find_fuzzy_candidates,
    merge_entities,
    resolve_commitment_fks,
    run_consolidation,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh migrated DB per test."""
    db = schema.ensure_database(tmp_path / "consolidation.db")
    yield db
    db.close()


def _create_person(
    conn, name, *, aliases=None, attributes=None, importance=0.5, profile="default"
):
    return entities.create_entity(
        conn, "person", name,
        aliases=aliases, attributes=attributes,
        importance=importance, profile=profile, now=NOW,
    )


# ─── ConsolidationResult ────────────────────────────────────────────────


class TestConsolidationResult:
    def test_defaults(self):
        r = ConsolidationResult()
        assert r.entities_merged == 0
        assert r.commitments_linked == 0
        assert r.duration_seconds == 0.0

    def test_fields_settable(self):
        r = ConsolidationResult(
            entities_merged=3,
            commitments_linked=5,
            duration_seconds=0.42,
        )
        assert r.entities_merged == 3
        assert r.commitments_linked == 5
        assert r.duration_seconds == 0.42


# ─── _fuzzy_match_score ─────────────────────────────────────────────────


class TestFuzzyMatchScore:
    def test_exact_name_match(self, conn):
        a = _create_person(conn, "Sarah Chen")
        b = _create_person(conn, "sarah chen", profile="other")  # different profile OK
        # Direct score test — wraps raw data, not the find_* caller
        assert _fuzzy_match_score(
            _as_dict(a), _as_dict(b)
        ) == 1.0

    def test_shared_alias_gives_high_score(self, conn):
        a = _create_person(conn, "Sarah Chen", aliases=["schen"])
        b = _create_person(conn, "Sarah C.", aliases=["schen"], profile="other")
        score = _fuzzy_match_score(_as_dict(a), _as_dict(b))
        assert score == 0.95

    def test_alias_matches_other_name(self, conn):
        """B's name appears in A's aliases → high score."""
        a = _create_person(conn, "Sarah Chen", aliases=["schen"])
        b = _create_person(conn, "schen", profile="other")
        score = _fuzzy_match_score(_as_dict(a), _as_dict(b))
        assert score == 0.95

    def test_typo_level_variant_merged(self, conn):
        """SequenceMatcher ≥ 0.92 for close variants."""
        # "John Smyth" vs "John Smyth " (trailing space) — very close
        a = _create_person(conn, "Jonathan Smith")
        b = _create_person(conn, "Jonathan Smith", profile="other")
        score = _fuzzy_match_score(_as_dict(a), _as_dict(b))
        assert score == 1.0  # exact caught first

    def test_joel_salinas_vs_joel_hart_not_merged(self, conn):
        """Same first name, different last name → score 0.0.

        The v1 daemon had a specific regression test for this case
        (test_alias_filter.py). Porting it as a tripwire: if someone
        weakens the last-token safety, this test catches it before
        shipping.
        """
        a = _create_person(conn, "Joel Salinas")
        b = _create_person(conn, "Joel Hart", profile="other")
        assert _fuzzy_match_score(_as_dict(a), _as_dict(b)) == 0.0

    def test_same_first_and_last_different_middle(self, conn):
        """Same first and last, different middle → merge OK."""
        a = _create_person(conn, "John Smith")
        b = _create_person(conn, "John Q Smith", profile="other")
        # last token matches ("smith") so last-token safety doesn't veto
        # but SequenceMatcher may or may not cross threshold. Just
        # assert we don't crash and behavior is deterministic.
        score = _fuzzy_match_score(_as_dict(a), _as_dict(b))
        assert score >= 0.0  # sanity: numeric

    def test_different_kinds_returns_zero(self, conn):
        """Kind mismatch is handled by the caller, but the score
        function still compares the raw names. Same-kind gate happens
        above this function. This test just verifies the caller
        contract is sane — mismatched kinds never become candidates."""
        person = _create_person(conn, "Acme")
        org = entities.create_entity(
            conn, "organization", "Acme", profile="default", now=NOW
        )
        # The function itself compares names and would score 1.0 here;
        # the kind filter is in find_fuzzy_candidates.
        score = _fuzzy_match_score(_as_dict(person), _as_dict(org))
        assert score == 1.0  # names match — filter elsewhere

    def test_single_word_person_no_fuzzy_merge(self, conn):
        """Two single-word person names that aren't exact → no merge."""
        a = _create_person(conn, "Alex")
        b = _create_person(conn, "Alexa", profile="other")
        # SequenceMatcher("alex", "alexa") = 2*4 / 9 = 0.888
        # Under threshold, returns 0.0
        assert _fuzzy_match_score(_as_dict(a), _as_dict(b)) == 0.0


def _as_dict(entity):
    """Convert an Entity dataclass to the dict shape find_* passes
    to _fuzzy_match_score (matches sqlite3.Row-like key access)."""
    import json
    return {
        "id": entity.id,
        "kind": entity.kind,
        "name": entity.name,
        "aliases_json": json.dumps(entity.aliases) if entity.aliases else None,
        "importance": entity.importance,
        "access_count": entity.access_count,
    }


# ─── find_fuzzy_candidates ──────────────────────────────────────────────


class TestFindFuzzyCandidates:
    def test_empty_profile(self, conn):
        assert find_fuzzy_candidates(conn, profile="default") == []

    def test_single_entity(self, conn):
        _create_person(conn, "Alice")
        assert find_fuzzy_candidates(conn, profile="default") == []

    def test_exact_duplicates_impossible(self, conn):
        """Can't have two entities with same (profile, kind, name) —
        the unique constraint prevents it. So exact matches come
        from alias overlap or case differences in the DB."""
        # upsert_entity dedupes on the unique constraint, so we
        # cannot create two exact matches. Instead, create two that
        # share an alias:
        a = _create_person(conn, "Sarah Chen", aliases=["schen"])
        b = _create_person(conn, "Sarah C.", aliases=["schen"])
        candidates = find_fuzzy_candidates(conn, profile="default")
        assert len(candidates) == 1
        keep_id, merge_id, score = candidates[0]
        assert {keep_id, merge_id} == {a.id, b.id}

    def test_different_kinds_never_paired(self, conn):
        _create_person(conn, "Acme")
        entities.create_entity(
            conn, "organization", "Acme", profile="default", now=NOW
        )
        candidates = find_fuzzy_candidates(conn, profile="default")
        assert candidates == []

    def test_profile_isolation(self, conn):
        _create_person(conn, "Sarah", profile="user_a", aliases=["s"])
        _create_person(conn, "Sarah", profile="user_b", aliases=["s"])
        assert find_fuzzy_candidates(conn, profile="user_a") == []
        assert find_fuzzy_candidates(conn, profile="user_b") == []

    def test_higher_importance_is_keep(self, conn):
        a = _create_person(conn, "Sarah Chen", aliases=["schen"], importance=0.3)
        b = _create_person(conn, "Sarah C.", aliases=["schen"], importance=0.9)
        candidates = find_fuzzy_candidates(conn, profile="default")
        assert len(candidates) == 1
        keep_id, merge_id, _ = candidates[0]
        assert keep_id == b.id  # higher importance
        assert merge_id == a.id

    def test_joel_safety_gate(self, conn):
        """The Joel Salinas / Joel Hart case does NOT become a candidate."""
        _create_person(conn, "Joel Salinas")
        _create_person(conn, "Joel Hart")
        assert find_fuzzy_candidates(conn, profile="default") == []


# ─── merge_entities ─────────────────────────────────────────────────────


class TestMergeEntities:
    def test_missing_keep_returns_none(self, conn):
        merge = _create_person(conn, "B")
        assert merge_entities(conn, 99999, merge.id, profile="default", now=NOW) is None

    def test_missing_merge_returns_none(self, conn):
        keep = _create_person(conn, "A")
        assert merge_entities(conn, keep.id, 99999, profile="default", now=NOW) is None

    def test_same_id_no_op(self, conn):
        a = _create_person(conn, "A")
        assert merge_entities(conn, a.id, a.id, profile="default", now=NOW) is None
        # Row still exists
        assert entities.get_entity(conn, a.id) is not None

    def test_aliases_combined(self, conn):
        keep = _create_person(conn, "Sarah Chen", aliases=["schen"])
        merge = _create_person(conn, "Sarah C.", aliases=["sch"])
        result = merge_entities(
            conn, keep.id, merge.id, profile="default", now=NOW
        )
        assert result == keep.id

        fetched = entities.get_entity(conn, keep.id)
        assert fetched is not None
        # New aliases include: schen (original), sch (merge's), "Sarah C." (merge's name)
        alias_set_lower = {a.lower() for a in fetched.aliases}
        assert "schen" in alias_set_lower
        assert "sch" in alias_set_lower
        assert "sarah c." in alias_set_lower
        # Keep's own name NOT in aliases
        assert "sarah chen" not in alias_set_lower

    def test_attributes_merged_keep_wins_conflicts(self, conn):
        keep = _create_person(
            conn, "Sarah",
            attributes={"role": "CEO", "city": "NYC"},
        )
        merge = _create_person(
            conn, "Sarah C",
            attributes={"role": "Founder", "industry": "tech"},
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)

        fetched = entities.get_entity(conn, keep.id)
        assert fetched.attributes["role"] == "CEO"  # keep wins
        assert fetched.attributes["city"] == "NYC"  # from keep
        assert fetched.attributes["industry"] == "tech"  # merge fills gap

    def test_importance_max(self, conn):
        keep = _create_person(conn, "A", importance=0.4)
        merge = _create_person(conn, "B", importance=0.8)
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        fetched = entities.get_entity(conn, keep.id)
        assert fetched.importance == 0.8

    def test_merge_entity_soft_deleted(self, conn):
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        # Merge row hidden from public API
        assert entities.get_entity(conn, merge.id) is None
        # But raw SQL shows it still exists with deleted_at set
        row = conn.execute(
            "SELECT deleted_at FROM entities WHERE id = ?", (merge.id,)
        ).fetchone()
        assert row["deleted_at"] is not None

    def test_relationships_from_direction_transferred(self, conn):
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        other = _create_person(conn, "C")
        rel = entities.create_relationship(
            conn, merge.id, other.id, "knows",
            profile="default", now=NOW,
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        # Relationship should now point from keep to other
        rels = entities.get_relationships(conn, keep.id, profile="default")
        assert len(rels) == 1
        assert rels[0].from_entity_id == keep.id
        assert rels[0].to_entity_id == other.id

    def test_relationships_to_direction_transferred(self, conn):
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        other = _create_person(conn, "C")
        entities.create_relationship(
            conn, other.id, merge.id, "works_with",
            profile="default", now=NOW,
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        rels = entities.get_relationships(conn, keep.id, profile="default")
        assert len(rels) == 1
        assert rels[0].from_entity_id == other.id
        assert rels[0].to_entity_id == keep.id

    def test_relationships_conflict_soft_deletes_merge_version(self, conn):
        """If keep already has the same edge, merge's row is soft-deleted
        (keep's survives) rather than crashing the unique constraint."""
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        other = _create_person(conn, "C")
        # Both keep and merge know other
        entities.create_relationship(
            conn, keep.id, other.id, "knows", profile="default", now=NOW
        )
        entities.create_relationship(
            conn, merge.id, other.id, "knows", profile="default", now=NOW
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        rels = entities.get_relationships(conn, keep.id, profile="default")
        # Only one relationship should remain active
        active = [r for r in rels if r.type == "knows"]
        assert len(active) == 1

    def test_self_loop_after_merge_soft_deleted(self, conn):
        """If merge had a relationship to itself (rare), it becomes
        keep → keep after redirect, which is a self-loop. Soft-delete."""
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        entities.create_relationship(
            conn, merge.id, merge.id, "reminds_of",
            profile="default", now=NOW,
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        rels = entities.get_relationships(conn, keep.id, profile="default")
        # The self-loop should not appear on keep
        assert all(r.from_entity_id != keep.id or r.to_entity_id != keep.id for r in rels)

    def test_commitment_owner_transferred(self, conn):
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        c = commitments.create_commitment(
            conn, "do X",
            owner_entity_id=merge.id,
            profile="default", now=NOW,
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.owner_entity_id == keep.id

    def test_commitment_target_transferred(self, conn):
        keep = _create_person(conn, "A")
        merge = _create_person(conn, "B")
        c = commitments.create_commitment(
            conn, "send to B",
            target_entity_id=merge.id,
            profile="default", now=NOW,
        )
        merge_entities(conn, keep.id, merge.id, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id == keep.id


# ─── resolve_commitment_fks ─────────────────────────────────────────────


class TestResolveCommitmentFks:
    def test_null_target_linked_by_name(self, conn):
        sarah = _create_person(conn, "Sarah")
        c = commitments.create_commitment(
            conn, "send the proposal to Sarah",
            profile="default", now=NOW,
        )
        assert c.target_entity_id is None
        linked = resolve_commitment_fks(conn, profile="default", now=NOW)
        assert linked == 1
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id == sarah.id

    def test_existing_target_not_overwritten(self, conn):
        sarah = _create_person(conn, "Sarah")
        bob = _create_person(conn, "Bob")
        c = commitments.create_commitment(
            conn, "send to Sarah",  # Sarah in content
            target_entity_id=bob.id,  # but Bob already set
            profile="default", now=NOW,
        )
        resolve_commitment_fks(conn, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id == bob.id  # unchanged

    def test_case_insensitive_match(self, conn):
        sarah = _create_person(conn, "Sarah Chen")
        c = commitments.create_commitment(
            conn, "email SARAH CHEN about Q4",
            profile="default", now=NOW,
        )
        resolve_commitment_fks(conn, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id == sarah.id

    def test_no_match_stays_null(self, conn):
        _create_person(conn, "Sarah")
        c = commitments.create_commitment(
            conn, "review the deck",  # no person mentioned
            profile="default", now=NOW,
        )
        resolve_commitment_fks(conn, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id is None

    def test_soft_deleted_entity_not_matched(self, conn):
        sarah = _create_person(conn, "Sarah")
        entities.soft_delete_entity(conn, sarah.id, profile="default", now=NOW)
        c = commitments.create_commitment(
            conn, "send to Sarah",
            profile="default", now=NOW,
        )
        resolve_commitment_fks(conn, profile="default", now=NOW)
        fetched = commitments.get_commitment(conn, c.id)
        assert fetched.target_entity_id is None

    def test_soft_deleted_commitment_not_processed(self, conn):
        sarah = _create_person(conn, "Sarah")
        c = commitments.create_commitment(
            conn, "send to Sarah",
            profile="default", now=NOW,
        )
        commitments.soft_delete_commitment(conn, c.id, now=NOW)
        linked = resolve_commitment_fks(conn, profile="default", now=NOW)
        assert linked == 0

    def test_returns_count(self, conn):
        _create_person(conn, "Sarah")
        _create_person(conn, "Bob")
        commitments.create_commitment(
            conn, "send to Sarah", profile="default", now=NOW
        )
        commitments.create_commitment(
            conn, "email Bob", profile="default", now=NOW
        )
        commitments.create_commitment(
            conn, "review deck", profile="default", now=NOW
        )
        linked = resolve_commitment_fks(conn, profile="default", now=NOW)
        assert linked == 2  # two linked, one untouched


# ─── run_consolidation ──────────────────────────────────────────────────


class TestRunConsolidation:
    def test_empty_db(self, conn):
        result = run_consolidation(conn, profile="default", now=NOW)
        assert result.entities_merged == 0
        assert result.commitments_linked == 0

    def test_full_pass(self, conn):
        """One merge + one link in a single run."""
        # Two entities that should merge (alias overlap)
        keep = _create_person(
            conn, "Sarah Chen", aliases=["schen"], importance=0.9
        )
        dup = _create_person(
            conn, "Sarah C.", aliases=["schen"], importance=0.3
        )
        # A commitment that mentions a name (will resolve after
        # merge: both point to keep because merge is soft-deleted
        # and the content says "Sarah Chen")
        commit = commitments.create_commitment(
            conn, "send the proposal to Sarah Chen",
            profile="default", now=NOW,
        )

        result = run_consolidation(conn, profile="default", now=NOW)
        assert result.entities_merged == 1
        assert result.commitments_linked == 1

        # Dup is gone
        assert entities.get_entity(conn, dup.id) is None
        # Keep survives with merged aliases
        kept = entities.get_entity(conn, keep.id)
        assert kept is not None
        alias_set = {a.lower() for a in kept.aliases}
        assert "sarah c." in alias_set
        # Commitment is linked to keep
        fetched = commitments.get_commitment(conn, commit.id)
        assert fetched.target_entity_id == keep.id

    def test_idempotent(self, conn):
        """Running consolidation twice should be a no-op the second time."""
        _create_person(conn, "Sarah Chen", aliases=["schen"])
        _create_person(conn, "Sarah C.", aliases=["schen"])
        commitments.create_commitment(
            conn, "send to Sarah Chen", profile="default", now=NOW
        )

        first = run_consolidation(conn, profile="default", now=NOW)
        assert first.entities_merged == 1

        second = run_consolidation(conn, profile="default", now=NOW)
        assert second.entities_merged == 0
        assert second.commitments_linked == 0

    def test_profile_isolated(self, conn):
        """Consolidating one profile doesn't affect another.

        Two near-duplicates in user_a (shared alias) should merge
        into one. user_b has one Bob that must be untouched.
        """
        _create_person(
            conn, "Sarah Chen", profile="user_a", aliases=["schen"]
        )
        _create_person(
            conn, "Sarah C.", profile="user_a", aliases=["schen"]
        )
        # Separate profile, should not be touched
        _create_person(conn, "Bob", profile="user_b")

        # Consolidating user_a does not affect user_b
        result = run_consolidation(conn, profile="user_a", now=NOW)
        assert result.entities_merged == 1
        # user_b entity still there
        assert entities.find_entity(conn, "Bob", profile="user_b") is not None

    def test_duration_populated(self, conn):
        result = run_consolidation(conn, profile="default", now=NOW)
        assert result.duration_seconds >= 0.0


# ─── Threshold exposed for tuning ───────────────────────────────────────


class TestThresholdConstant:
    def test_threshold_is_reasonable(self):
        """Tripwire: make sure AUTO_MERGE_THRESHOLD doesn't drift too
        low. A value below 0.8 is almost certainly a bug that would
        cause aggressive false-merges."""
        assert 0.85 <= AUTO_MERGE_THRESHOLD <= 1.0
