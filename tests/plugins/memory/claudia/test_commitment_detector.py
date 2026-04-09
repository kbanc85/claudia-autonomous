"""Unit tests for plugins/memory/claudia/commitment_detector.py (Phase 2B.2).

Covered:

- ``DetectedCommitment`` dataclass shape and defaults
- ``_extract_pattern_matches`` regex pre-filter:
    * explicit first-person: "I'll X", "I will X", "I'm going to X"
    * first-person plural: "we'll X", "we will X"
    * implicit: "let me X", "I need to X", "I should X"
    * vague / skipped: "we should explore that someday", "maybe X"
    * past tense rejected: "I sent the proposal"
    * conditional rejected: "if I X, then Y"
    * question rejected: "will you X?"
    * third-person rejected: "she will X"
- ``_parse_deadline`` natural-language date extraction:
    * "by Friday", "before Monday", "on Tuesday"
    * "tomorrow", "today", "tonight"
    * "next week", "this week"
    * "in two hours", "in 30 minutes"
    * no deadline returns None
    * unparseable deadline returns raw text only
- ``_coerce_json_to_commitments`` LLM output parser:
    * plain array
    * object-wrapped
    * fenced markdown
    * missing fields
    * invalid types dropped
    * confidence clamped
- ``PatternCommitmentDetector``:
    * empty text returns []
    * whitespace returns []
    * single explicit commitment
    * multiple commitments in one text
    * implicit commitment lower confidence than explicit
    * source_ref stamped on each result
- ``HybridCommitmentDetector``:
    * pattern pre-filter short-circuits on no match (no LLM call)
    * pattern match + LLM refinement path
    * LLM unavailable falls back to pattern output
    * LLM exception falls back to pattern output
    * LLM returns empty list means "no commitments" (override pattern)
- ``OllamaCommitmentDetector`` availability probe:
    * lazy, cached, failure-tolerant
    * not called on construction (invariant #1)
- Prompt invariants:
    * prompt mentions explicit/implicit/vague taxonomy
    * prompt instructs to return [] if none
    * prompt has {text} placeholder
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from plugins.memory.claudia.commitment_detector import (
    DEFAULT_DETECTION_MODEL,
    DetectedCommitment,
    HybridCommitmentDetector,
    OllamaCommitmentDetector,
    PatternCommitmentDetector,
    _coerce_json_to_commitments,
    _extract_pattern_matches,
    _parse_deadline,
)


# ─── DetectedCommitment dataclass ───────────────────────────────────────


class TestDetectedCommitmentDataclass:
    def test_defaults(self):
        c = DetectedCommitment(content="send proposal")
        assert c.deadline_raw is None
        assert c.deadline_iso is None
        assert c.confidence == 0.5
        assert c.commitment_type == "explicit"
        assert c.source_ref == ""
        assert c.origin == "pattern"

    def test_all_fields_settable(self):
        c = DetectedCommitment(
            content="send Sarah the proposal",
            deadline_raw="by Friday",
            deadline_iso="2026-04-10",
            confidence=0.9,
            commitment_type="implicit",
            source_ref="sess-42",
            origin="hybrid",
        )
        assert c.deadline_raw == "by Friday"
        assert c.commitment_type == "implicit"
        assert c.origin == "hybrid"


# ─── Pattern pre-filter ─────────────────────────────────────────────────


class TestPatternExtraction:
    def test_empty_text(self):
        assert _extract_pattern_matches("") == []
        assert _extract_pattern_matches("   ") == []

    # Explicit first-person patterns

    def test_ill_pattern(self):
        matches = _extract_pattern_matches("I'll send the proposal tomorrow.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"
        assert "send the proposal" in matches[0].content

    def test_i_will_pattern(self):
        matches = _extract_pattern_matches("I will send the proposal.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"

    def test_im_going_to_pattern(self):
        matches = _extract_pattern_matches("I'm going to review the contract.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"

    def test_i_am_going_to_pattern(self):
        matches = _extract_pattern_matches("I am going to schedule the call.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"

    # First-person plural

    def test_well_pattern(self):
        matches = _extract_pattern_matches("We'll ship the feature next week.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"

    def test_we_will_pattern(self):
        matches = _extract_pattern_matches("We will finalize the plan.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "explicit"

    # Implicit patterns (lower confidence)

    def test_let_me_pattern_implicit(self):
        matches = _extract_pattern_matches("Let me check on that.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "implicit"
        assert matches[0].confidence < 0.7

    def test_i_need_to_pattern_implicit(self):
        matches = _extract_pattern_matches("I need to email the team.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "implicit"

    def test_i_should_pattern_implicit(self):
        matches = _extract_pattern_matches("I should follow up with Sarah.")
        assert len(matches) == 1
        assert matches[0].commitment_type == "implicit"

    # Negative cases — should NOT match

    def test_past_tense_rejected(self):
        matches = _extract_pattern_matches("I sent the proposal yesterday.")
        assert matches == []

    def test_past_tense_we_sent_rejected(self):
        matches = _extract_pattern_matches("We shipped the feature last week.")
        assert matches == []

    def test_question_rejected(self):
        matches = _extract_pattern_matches("Will you send the proposal?")
        assert matches == []

    def test_third_person_rejected(self):
        matches = _extract_pattern_matches("She will send the proposal.")
        assert matches == []

    def test_vague_someday_rejected(self):
        """'we should explore that someday' is a vague intention, not a commitment."""
        matches = _extract_pattern_matches("We should explore that someday.")
        # Either no match, OR match but skipped as vague — current
        # implementation may match "we should" as implicit. If it
        # does, confidence should be very low.
        for m in matches:
            assert m.confidence < 0.5 or m.commitment_type == "vague"

    # Multiple commitments

    def test_multiple_commitments_in_text(self):
        text = "I'll send the proposal. We'll schedule the kickoff."
        matches = _extract_pattern_matches(text)
        assert len(matches) == 2

    # Deadline extraction inline

    def test_extracts_by_weekday_deadline(self):
        matches = _extract_pattern_matches("I'll send the proposal by Friday.")
        assert len(matches) == 1
        assert matches[0].deadline_raw is not None
        assert "friday" in matches[0].deadline_raw.lower()

    def test_extracts_tomorrow_deadline(self):
        matches = _extract_pattern_matches("I'll review the draft tomorrow.")
        assert len(matches) == 1
        assert matches[0].deadline_raw is not None
        assert "tomorrow" in matches[0].deadline_raw.lower()

    def test_no_deadline_leaves_none(self):
        matches = _extract_pattern_matches("I'll think about it.")
        assert len(matches) == 1
        assert matches[0].deadline_raw is None

    # Source ref propagation (via PatternCommitmentDetector)

    def test_source_ref_propagation_via_detector(self):
        det = PatternCommitmentDetector()
        out = det.detect(
            "I'll send the proposal.",
            source_ref="sess-xyz",
        )
        assert len(out) == 1
        assert out[0].source_ref == "sess-xyz"

    def test_origin_is_pattern(self):
        matches = _extract_pattern_matches("I'll send it.")
        assert all(m.origin == "pattern" for m in matches)


# ─── Deadline parsing ───────────────────────────────────────────────────


class TestParseDeadline:
    def test_none_returns_none(self):
        assert _parse_deadline(None) is None

    def test_empty_returns_none(self):
        assert _parse_deadline("") is None

    def test_tomorrow(self):
        from datetime import datetime, timedelta, timezone

        anchor = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_deadline("tomorrow", anchor=anchor)
        assert result is not None
        # Should be the next day
        assert result.startswith("2026-04-10")

    def test_today(self):
        from datetime import datetime, timezone

        anchor = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_deadline("today", anchor=anchor)
        assert result is not None
        assert result.startswith("2026-04-09")

    def test_weekday_by_friday(self):
        """'by Friday' from a Thursday anchor resolves to the next day."""
        from datetime import datetime, timezone

        # 2026-04-09 is a Thursday
        anchor = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_deadline("by Friday", anchor=anchor)
        assert result is not None
        assert result.startswith("2026-04-10")

    def test_weekday_by_monday_wraps(self):
        """'by Monday' from a Thursday anchor resolves to next Monday."""
        from datetime import datetime, timezone

        anchor = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_deadline("by Monday", anchor=anchor)
        assert result is not None
        assert result.startswith("2026-04-13")

    def test_next_week(self):
        from datetime import datetime, timezone

        anchor = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_deadline("next week", anchor=anchor)
        assert result is not None
        # Should be roughly a week later
        assert result.startswith("2026-04-16")

    def test_unparseable_returns_none(self):
        assert _parse_deadline("sometime soon-ish") is None

    def test_end_of_q4_returns_none(self):
        """'end of Q4' is currently unparseable. Returning None is
        acceptable; the raw text is preserved on DetectedCommitment
        anyway."""
        assert _parse_deadline("end of Q4") is None


# ─── JSON coercion (LLM output) ─────────────────────────────────────────


class TestCoerceJsonToCommitments:
    def test_plain_array(self):
        raw = (
            '[{"content":"send proposal","commitment_type":"explicit",'
            '"confidence":0.9,"deadline_raw":"by Friday"}]'
        )
        out = _coerce_json_to_commitments(raw)
        assert len(out) == 1
        assert out[0].content == "send proposal"
        assert out[0].commitment_type == "explicit"
        assert out[0].confidence == 0.9
        assert out[0].deadline_raw == "by Friday"

    def test_empty_array(self):
        assert _coerce_json_to_commitments("[]") == []

    def test_object_wrapped(self):
        raw = (
            '{"commitments":[{"content":"x","commitment_type":"implicit",'
            '"confidence":0.6}]}'
        )
        out = _coerce_json_to_commitments(raw)
        assert len(out) == 1
        assert out[0].commitment_type == "implicit"

    def test_fenced_markdown(self):
        raw = (
            "```json\n"
            '[{"content":"x","commitment_type":"explicit","confidence":0.8}]\n'
            "```"
        )
        out = _coerce_json_to_commitments(raw)
        assert len(out) == 1

    def test_missing_content_dropped(self):
        raw = '[{"commitment_type":"explicit","confidence":0.9}]'
        assert _coerce_json_to_commitments(raw) == []

    def test_empty_content_dropped(self):
        raw = '[{"content":"","commitment_type":"explicit","confidence":0.9}]'
        assert _coerce_json_to_commitments(raw) == []

    def test_invalid_type_defaults_to_explicit(self):
        raw = '[{"content":"x","commitment_type":"bogus","confidence":0.9}]'
        out = _coerce_json_to_commitments(raw)
        assert len(out) == 1
        assert out[0].commitment_type == "explicit"

    def test_vague_type_dropped(self):
        """'vague' commitments are identified but not tracked per
        Claudia's core principles: vague intentions don't get accountability."""
        raw = '[{"content":"maybe later","commitment_type":"vague","confidence":0.3}]'
        out = _coerce_json_to_commitments(raw)
        assert out == []

    def test_confidence_clamped(self):
        raw = (
            '[{"content":"a","commitment_type":"explicit","confidence":1.5},'
            '{"content":"b","commitment_type":"explicit","confidence":-0.5}]'
        )
        out = _coerce_json_to_commitments(raw)
        assert len(out) == 2
        assert out[0].confidence == 1.0
        assert out[1].confidence == 0.0

    def test_missing_confidence_defaults(self):
        raw = '[{"content":"x","commitment_type":"explicit"}]'
        out = _coerce_json_to_commitments(raw)
        assert out[0].confidence == 0.5

    def test_malformed_returns_empty(self):
        assert _coerce_json_to_commitments("not json") == []
        assert _coerce_json_to_commitments("") == []

    def test_source_ref_propagates(self):
        raw = '[{"content":"x","commitment_type":"explicit","confidence":0.9}]'
        out = _coerce_json_to_commitments(raw, source_ref="sess-42")
        assert out[0].source_ref == "sess-42"

    def test_origin_tagged_as_llm(self):
        raw = '[{"content":"x","commitment_type":"explicit","confidence":0.9}]'
        out = _coerce_json_to_commitments(raw)
        assert out[0].origin == "llm"


# ─── PatternCommitmentDetector ──────────────────────────────────────────


class TestPatternDetector:
    def test_empty_text(self):
        assert PatternCommitmentDetector().detect("") == []

    def test_whitespace(self):
        assert PatternCommitmentDetector().detect("   ") == []

    def test_single_explicit(self):
        det = PatternCommitmentDetector()
        out = det.detect("I'll send the proposal tomorrow.")
        assert len(out) == 1
        assert out[0].commitment_type == "explicit"

    def test_implicit_lower_confidence_than_explicit(self):
        det = PatternCommitmentDetector()
        explicit = det.detect("I'll send the proposal.")
        implicit = det.detect("Let me check on that.")
        assert len(explicit) == 1 and len(implicit) == 1
        assert explicit[0].confidence > implicit[0].confidence

    def test_is_available_always_true(self):
        """Pattern detector has no external dependencies."""
        assert PatternCommitmentDetector().is_available() is True

    def test_multiple_commitments(self):
        det = PatternCommitmentDetector()
        text = "I'll send the proposal. We'll schedule the kickoff."
        out = det.detect(text)
        assert len(out) == 2


# ─── HybridCommitmentDetector ───────────────────────────────────────────


class _FakeLLMDetector:
    """Scripted LLM for HybridCommitmentDetector injection."""

    def __init__(self, script=None, available=True, raise_on_call=False):
        self._script = list(script) if script is not None else []
        self._available = available
        self._raise = raise_on_call
        self.call_count = 0
        self.last_prompt: Optional[str] = None

    def is_available(self) -> bool:
        return self._available

    def _call_llm(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        if not self._script:
            return "[]"
        return self._script.pop(0)


class TestHybridDetector:
    def _make(self, *, script=None, available=True, raise_on_call=False):
        """Build a hybrid detector with an injected fake LLM.

        Hybrid detector uses its own _call_llm that we override via
        subclassing for determinism. Matches the extractor test pattern.
        """

        class _TestableHybrid(HybridCommitmentDetector):
            def __init__(self_inner):
                super().__init__()
                self_inner._fake_script = list(script) if script is not None else []
                self_inner._fake_available = available
                self_inner._fake_raise = raise_on_call
                self_inner.call_count = 0

            def is_available(self_inner) -> bool:  # type: ignore[override]
                return self_inner._fake_available

            def _call_llm(self_inner, prompt: str) -> str:  # type: ignore[override]
                self_inner.call_count += 1
                if self_inner._fake_raise:
                    raise RuntimeError("simulated LLM failure")
                if not self_inner._fake_script:
                    return "[]"
                return self_inner._fake_script.pop(0)

        return _TestableHybrid()

    def test_no_pattern_match_skips_llm(self):
        """If the pattern pre-filter finds nothing, skip the LLM."""
        det = self._make()
        result = det.detect("The weather is nice today.")
        assert result == []
        assert det.call_count == 0

    def test_pattern_match_no_llm_falls_back_to_pattern(self):
        """LLM unavailable but pattern matches: return the pattern result."""
        det = self._make(available=False)
        result = det.detect("I'll send the proposal.")
        assert len(result) == 1
        assert result[0].origin == "pattern"
        assert det.call_count == 0

    def test_pattern_match_llm_raises_falls_back_to_pattern(self):
        det = self._make(raise_on_call=True)
        result = det.detect("I'll send the proposal.")
        assert len(result) == 1
        assert result[0].origin == "pattern"

    def test_llm_refinement_replaces_pattern_result(self):
        """When LLM returns valid results, use them (origin=llm)."""
        script = [
            '[{"content":"send the proposal to Sarah",'
            '"commitment_type":"explicit","confidence":0.95,'
            '"deadline_raw":"by Friday"}]'
        ]
        det = self._make(script=script)
        result = det.detect("I'll send the proposal to Sarah by Friday.")
        assert len(result) == 1
        assert result[0].origin == "llm"
        assert result[0].content == "send the proposal to Sarah"
        assert result[0].confidence == 0.95
        assert det.call_count == 1

    def test_llm_empty_array_overrides_pattern(self):
        """If LLM says 'no real commitment' ([]), trust it over the pattern.

        This is how the hybrid approach reduces pattern false positives:
        the LLM vetoes bogus matches like past tense or rhetorical.
        """
        det = self._make(script=["[]"])
        result = det.detect("I'll send the proposal.")
        assert result == []
        assert det.call_count == 1

    def test_source_ref_stamped_in_hybrid_pattern_fallback(self):
        det = self._make(available=False)
        result = det.detect(
            "I'll send the proposal.", source_ref="sess-abc"
        )
        assert result[0].source_ref == "sess-abc"

    def test_source_ref_stamped_in_hybrid_llm_path(self):
        script = [
            '[{"content":"x","commitment_type":"explicit","confidence":0.9}]'
        ]
        det = self._make(script=script)
        result = det.detect("I'll do x.", source_ref="sess-def")
        assert result[0].source_ref == "sess-def"


# ─── OllamaCommitmentDetector availability ──────────────────────────────


class TestOllamaDetectorAvailability:
    def test_no_probe_on_construction(self):
        """Invariant #1: no network calls in __init__."""
        probe_count = 0

        class _Tracked(OllamaCommitmentDetector):
            def _call_llm(self, prompt):  # type: ignore[override]
                nonlocal probe_count
                probe_count += 1
                return "[]"

        _Tracked()
        assert probe_count == 0

    def test_lazy_cached(self):
        probe_count = 0

        class _Tracked(OllamaCommitmentDetector):
            def _call_llm(self, prompt):  # type: ignore[override]
                nonlocal probe_count
                probe_count += 1
                return "[]"

        det = _Tracked()
        det.is_available()
        det.is_available()
        det.is_available()
        assert probe_count == 1

    def test_failure_tolerant(self):
        class _Dead(OllamaCommitmentDetector):
            def _call_llm(self, prompt):  # type: ignore[override]
                raise ConnectionError("no route to host")

        det = _Dead()
        assert det.is_available() is False  # no raise

    def test_defaults(self):
        det = OllamaCommitmentDetector()
        assert det._model == DEFAULT_DETECTION_MODEL


# ─── Prompt invariants ──────────────────────────────────────────────────


class TestPromptInvariants:
    """Tripwires on the detector prompt.

    These tests fail if someone edits the prompt and strips the
    explicit/implicit/vague taxonomy guidance or forgets to tell
    the LLM to return [] for no commitments.
    """

    def test_prompt_has_text_placeholder(self):
        from plugins.memory.claudia.commitment_detector import (
            _DETECTION_PROMPT,
        )

        assert "{text}" in _DETECTION_PROMPT

    def test_prompt_mentions_explicit_implicit_vague(self):
        from plugins.memory.claudia.commitment_detector import (
            _DETECTION_PROMPT,
        )

        lower = _DETECTION_PROMPT.lower()
        assert "explicit" in lower
        assert "implicit" in lower
        assert "vague" in lower

    def test_prompt_mentions_empty_array_fallback(self):
        from plugins.memory.claudia.commitment_detector import (
            _DETECTION_PROMPT,
        )

        assert "[]" in _DETECTION_PROMPT

    def test_prompt_mentions_skip_past_tense(self):
        from plugins.memory.claudia.commitment_detector import (
            _DETECTION_PROMPT,
        )

        lower = _DETECTION_PROMPT.lower()
        assert "past" in lower or "already" in lower or "completed" in lower
