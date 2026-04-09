"""Integration tests for ClaudiaMemoryProvider.consolidate (Phase 2B.3).

Exercises the end-to-end path:

    (populate entities via writer)
    -> provider.consolidate()
    -> provider.flush() implicitly (waits for any in-flight
       cognitive futures)
    -> writer runs consolidation.run_consolidation in a single
       transaction
    -> merged entities soft-deleted, commitments FK-resolved
    -> ConsolidationResult returned to the caller

Tests lock in:

- consolidate() returns a ConsolidationResult with correct counts
- Merging happens through the writer queue (flush before/after
  works)
- consolidate() works after real sync_turn calls that populated
  entities via the extraction path
- consolidate() is idempotent (second call is a no-op)
- consolidate() returns empty result on shutdown provider
- consolidate() uses the provider's profile (not hardcoded default)
- No Phase 2A/2B.1/2B.2 regressions (sync_turn still works
  alongside consolidation calls)
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from plugins.memory.claudia import (
    ClaudiaMemoryProvider,
    commitments as commitments_module,
    entities as entities_module,
)
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
)
from plugins.memory.claudia.consolidation import ConsolidationResult
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import ExtractedEntity, LLMExtractor


# ─── Test doubles ───────────────────────────────────────────────────────


class _FakeEmbedder(OllamaEmbedder):
    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        return [0.1, 0.2, 0.3]


class _ScriptedExtractor(LLMExtractor):
    """Extractor that returns a scripted list per call."""

    def __init__(self, script: Optional[List[List[ExtractedEntity]]] = None):
        self._script = list(script) if script is not None else []
        self.call_count = 0

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            return []
        return self._script.pop(0)


class _ScriptedDetector(CommitmentDetector):
    """Commitment detector that returns a scripted list per call."""

    def __init__(self, script: Optional[List[List[DetectedCommitment]]] = None):
        self._script = list(script) if script is not None else []
        self.call_count = 0

    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            return []
        results = self._script.pop(0)
        for c in results:
            if not c.source_ref:
                c.source_ref = source_ref
        return results


class _NoOpExtractor(LLMExtractor):
    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _NoOpDetector(CommitmentDetector):
    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        return []


class _TestProvider(ClaudiaMemoryProvider):
    """Provider with injectable extractor and detector."""

    def __init__(
        self,
        extractor: Optional[LLMExtractor] = None,
        detector: Optional[CommitmentDetector] = None,
    ):
        super().__init__()
        self._injected_extractor = extractor or _NoOpExtractor()
        self._injected_detector = detector or _NoOpDetector()

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._injected_extractor

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return self._injected_detector


def _provider(
    tmp_path,
    *,
    extractor: Optional[LLMExtractor] = None,
    detector: Optional[CommitmentDetector] = None,
    **init_kwargs,
) -> _TestProvider:
    p = _TestProvider(extractor=extractor, detector=detector)
    defaults = {"claudia_home": str(tmp_path), "platform": "cli"}
    defaults.update(init_kwargs)
    p.initialize(session_id="test-session", **defaults)
    return p


# ─── Helpers ────────────────────────────────────────────────────────────


def _seed_entity(p, name, *, kind="person", aliases=None, importance=0.5):
    """Create an entity via the writer queue (matches production path)."""

    def _job(conn):
        return entities_module.create_entity(
            conn, kind, name,
            aliases=aliases,
            importance=importance,
            profile=p._profile,
        )

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


def _seed_commitment(p, content, *, target_entity_id=None):
    def _job(conn):
        return commitments_module.create_commitment(
            conn,
            content,
            target_entity_id=target_entity_id,
            profile=p._profile,
        )

    return p._writer.enqueue_and_wait(_job, timeout=5.0)


def _count_entities(p):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM entities WHERE deleted_at IS NULL"
        ).fetchone()
    return row["n"]


def _count_commitments_with_target(p):
    assert p.flush(timeout=5.0)
    with p._reader_pool.acquire() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM commitments "
            "WHERE target_entity_id IS NOT NULL AND deleted_at IS NULL"
        ).fetchone()
    return row["n"]


# ─── Tests ──────────────────────────────────────────────────────────────


class TestConsolidateBasic:
    def test_empty_db_returns_zeros(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = p.consolidate(timeout=5.0)
            assert isinstance(result, ConsolidationResult)
            assert result.entities_merged == 0
            assert result.commitments_linked == 0
        finally:
            p.shutdown()

    def test_merges_duplicate_entities(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah Chen", aliases=["schen"], importance=0.9)
            _seed_entity(p, "Sarah C.", aliases=["schen"], importance=0.3)
            assert _count_entities(p) == 2

            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 1
            assert _count_entities(p) == 1
        finally:
            p.shutdown()

    def test_links_unlinked_commitments(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah")
            _seed_commitment(p, "send the proposal to Sarah")
            assert _count_commitments_with_target(p) == 0

            result = p.consolidate(timeout=5.0)
            assert result.commitments_linked == 1
            assert _count_commitments_with_target(p) == 1
        finally:
            p.shutdown()

    def test_full_pass_merges_and_links(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(
                p, "Sarah Chen", aliases=["schen"], importance=0.9
            )
            _seed_entity(
                p, "Sarah C.", aliases=["schen"], importance=0.3
            )
            _seed_commitment(p, "send proposal to Sarah Chen")

            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 1
            assert result.commitments_linked == 1
        finally:
            p.shutdown()

    def test_duration_populated(self, tmp_path):
        p = _provider(tmp_path)
        try:
            result = p.consolidate(timeout=5.0)
            assert result.duration_seconds >= 0.0
        finally:
            p.shutdown()


class TestConsolidateIdempotency:
    def test_second_call_no_op(self, tmp_path):
        p = _provider(tmp_path)
        try:
            _seed_entity(p, "Sarah Chen", aliases=["schen"])
            _seed_entity(p, "Sarah C.", aliases=["schen"])

            first = p.consolidate(timeout=5.0)
            assert first.entities_merged == 1

            second = p.consolidate(timeout=5.0)
            assert second.entities_merged == 0
            assert second.commitments_linked == 0
        finally:
            p.shutdown()


class TestConsolidateAfterSyncTurn:
    def test_consolidate_after_extraction_merges_related_entities(self, tmp_path):
        """Simulates real usage: sync_turn populates entities via
        extraction, then consolidate runs and finds duplicates."""
        extractor = _ScriptedExtractor(
            script=[
                [
                    ExtractedEntity(
                        name="Sarah Chen",
                        kind="person",
                        canonical_name="sarah chen",
                        confidence=0.9,
                        aliases=["schen"],
                    )
                ],
                [
                    ExtractedEntity(
                        name="Sarah C.",
                        kind="person",
                        canonical_name="sarah c.",
                        confidence=0.8,
                        aliases=["schen"],
                    )
                ],
            ]
        )
        p = _provider(tmp_path, extractor=extractor)
        try:
            p.sync_turn("I spoke to Sarah Chen", "ok")
            p.sync_turn("I spoke to Sarah C.", "ok")
            assert p.flush(timeout=5.0)
            assert _count_entities(p) == 2

            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 1
            assert _count_entities(p) == 1
        finally:
            p.shutdown()

    def test_consolidate_after_detection_links_commitments(self, tmp_path):
        """sync_turn populates a commitment. consolidate finds the
        entity (seeded separately) and links the commitment."""
        detector = _ScriptedDetector(
            script=[
                [
                    DetectedCommitment(
                        content="send the proposal to Sarah",
                        confidence=0.9,
                        commitment_type="explicit",
                    )
                ]
            ]
        )
        p = _provider(tmp_path, detector=detector)
        try:
            # Seed entity first (simulating prior turn's extraction)
            _seed_entity(p, "Sarah")

            # Detection run via sync_turn
            p.sync_turn("I'll send the proposal to Sarah", "ok")
            assert p.flush(timeout=5.0)
            assert _count_commitments_with_target(p) == 0

            # Consolidate links the commitment to the seeded entity
            result = p.consolidate(timeout=5.0)
            assert result.commitments_linked == 1
            assert _count_commitments_with_target(p) == 1
        finally:
            p.shutdown()


class TestConsolidateShutdown:
    def test_returns_empty_after_shutdown(self, tmp_path):
        p = _provider(tmp_path)
        p.shutdown()
        result = p.consolidate(timeout=5.0)
        assert result.entities_merged == 0
        assert result.commitments_linked == 0


class TestConsolidateProfileIsolation:
    def test_respects_provider_profile(self, tmp_path):
        """Consolidation runs against provider._profile, not a
        hardcoded default."""
        p = _provider(tmp_path, user_id="custom_user")
        try:
            assert p._profile == "custom_user"

            _seed_entity(
                p, "Sarah Chen", aliases=["schen"]
            )
            _seed_entity(
                p, "Sarah C.", aliases=["schen"]
            )

            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 1
        finally:
            p.shutdown()

    def test_does_not_touch_other_profile(self, tmp_path):
        """Seed entities in a non-provider profile; consolidate must
        not touch them even though their profile also has duplicates."""
        p = _provider(tmp_path, user_id="profile_a")
        try:
            # Seed duplicates in profile_a (the provider's profile)
            _seed_entity(p, "Sarah Chen", aliases=["schen"])
            _seed_entity(p, "Sarah C.", aliases=["schen"])

            # Seed duplicates directly in profile_b via raw SQL/writer
            def _seed_other(conn):
                entities_module.create_entity(
                    conn, "person", "Bob A",
                    aliases=["bob-a"], profile="profile_b",
                )
                entities_module.create_entity(
                    conn, "person", "Bob B.",
                    aliases=["bob-a"], profile="profile_b",
                )

            p._writer.enqueue_and_wait(_seed_other, timeout=5.0)

            # Consolidate only profile_a
            result = p.consolidate(timeout=5.0)
            assert result.entities_merged == 1

            # profile_b's duplicates are still both there
            with p._reader_pool.acquire() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM entities "
                    "WHERE profile = 'profile_b' AND deleted_at IS NULL"
                ).fetchone()
            assert row["n"] == 2
        finally:
            p.shutdown()
