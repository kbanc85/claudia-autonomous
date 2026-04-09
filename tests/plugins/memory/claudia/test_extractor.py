"""Unit tests for plugins/memory/claudia/extractor.py (Phase 2B.1).

Covers:

- ``ExtractedEntity`` dataclass defaults and field population
- ``_coerce_json_to_entities`` tolerant parsing:
    * plain JSON array
    * object-wrapped (``{"entities": [...]}``, ``{"results": [...]}``)
    * fenced markdown code blocks (```json ... ```)
    * non-JSON text with embedded JSON
    * missing required fields (dropped)
    * invalid ``kind`` (dropped)
    * out-of-range confidence (clamped to [0, 1])
    * aliases/attributes shape coercion
    * empty / malformed input
- ``OllamaLLMExtractor`` availability probe:
    * lazy — no probe until first call
    * cached after first call
    * returns False on connection error without raising
- ``OllamaLLMExtractor.extract``:
    * empty / whitespace text returns []
    * unavailable backend returns []
    * HTTP/parse errors swallowed, return []
    * happy path returns parsed entities with source_ref propagated
- Prompt invariants:
    * prompt explicitly rules out temporal expressions (days, months)
    * prompt explicitly rules out common nouns
    * prompt template is deterministic for identical inputs

Tests use a ``_FakeOllamaExtractor`` subclass that overrides
``_call_llm`` and ``is_available`` to return scripted responses,
mirroring the ``_FakeEmbedder`` pattern from Phase 2A test files.
No real HTTP calls are made.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from plugins.memory.claudia.extractor import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_OLLAMA_HOST,
    VALID_ENTITY_KINDS,
    ExtractedEntity,
    LLMExtractor,
    OllamaLLMExtractor,
    _coerce_json_to_entities,
)


# ─── Scripted fake extractor ────────────────────────────────────────────


class _FakeOllamaExtractor(OllamaLLMExtractor):
    """OllamaLLMExtractor with scripted ``_call_llm`` and forced availability.

    Pass ``script`` as a list of (raw_response_or_exception) to drive
    sequential calls. Omit ``script`` to return an empty array on every call.
    Set ``available=False`` to simulate an unreachable backend.
    """

    def __init__(
        self,
        script: Optional[List] = None,
        *,
        available: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._script = list(script) if script is not None else None
        self._forced_available = available
        self.call_count = 0

    def is_available(self) -> bool:  # type: ignore[override]
        return self._forced_available

    def _call_llm(self, prompt: str) -> str:  # type: ignore[override]
        self.call_count += 1
        if self._script is None:
            return "[]"
        if not self._script:
            raise AssertionError(
                f"_call_llm script exhausted; prompt was {prompt[:60]!r}"
            )
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# ─── ExtractedEntity dataclass ──────────────────────────────────────────


class TestExtractedEntity:
    def test_defaults_populated(self):
        ent = ExtractedEntity(
            name="Sarah Chen",
            kind="person",
            canonical_name="sarah chen",
            confidence=0.9,
        )
        assert ent.aliases == []
        assert ent.attributes == {}
        assert ent.source_ref == ""

    def test_all_fields_settable(self):
        ent = ExtractedEntity(
            name="Acme Corp",
            kind="organization",
            canonical_name="acme corp",
            confidence=0.7,
            aliases=["Acme", "ACME"],
            attributes={"industry": "tech"},
            source_ref="session-123",
        )
        assert ent.aliases == ["Acme", "ACME"]
        assert ent.attributes == {"industry": "tech"}
        assert ent.source_ref == "session-123"

    def test_each_aliases_and_attributes_are_independent(self):
        """Default factory avoids the mutable-default-argument trap."""
        a = ExtractedEntity(name="A", kind="person", canonical_name="a", confidence=0.5)
        b = ExtractedEntity(name="B", kind="person", canonical_name="b", confidence=0.5)
        a.aliases.append("A1")
        a.attributes["k"] = "v"
        assert b.aliases == []
        assert b.attributes == {}

    def test_valid_entity_kinds_matches_entities_module(self):
        """VALID_ENTITY_KINDS must match entities.VALID_KINDS.

        If these drift, upsert_entity will reject valid extractor
        output with ``ValueError: invalid kind``. Keep them locked.
        """
        from plugins.memory.claudia.entities import VALID_KINDS

        assert VALID_ENTITY_KINDS == VALID_KINDS


# ─── JSON coercion ──────────────────────────────────────────────────────


class TestCoerceJsonPlainArray:
    def test_well_formed_array(self):
        raw = (
            '[{"name":"Sarah Chen","kind":"person",'
            '"canonical_name":"sarah chen","confidence":0.9}]'
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].name == "Sarah Chen"
        assert out[0].kind == "person"
        assert out[0].canonical_name == "sarah chen"
        assert out[0].confidence == 0.9

    def test_multiple_entities_in_array(self):
        raw = (
            '[{"name":"Sarah","kind":"person","canonical_name":"sarah","confidence":0.9},'
            '{"name":"Acme","kind":"organization","canonical_name":"acme","confidence":0.8}]'
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 2
        assert {e.kind for e in out} == {"person", "organization"}

    def test_empty_array(self):
        assert _coerce_json_to_entities("[]") == []

    def test_source_ref_propagates(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":0.5}]'
        out = _coerce_json_to_entities(raw, source_ref="sess-42")
        assert out[0].source_ref == "sess-42"


class TestCoerceJsonObjectWrapped:
    def test_entities_key(self):
        raw = (
            '{"entities":[{"name":"A","kind":"person",'
            '"canonical_name":"a","confidence":0.6}]}'
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].name == "A"

    def test_results_key(self):
        raw = (
            '{"results":[{"name":"B","kind":"project",'
            '"canonical_name":"b","confidence":0.7}]}'
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].kind == "project"

    def test_object_without_known_key_returns_empty(self):
        raw = '{"stuff":[{"name":"A","kind":"person","canonical_name":"a","confidence":0.5}]}'
        out = _coerce_json_to_entities(raw)
        assert out == []


class TestCoerceJsonFencedMarkdown:
    def test_json_fence(self):
        raw = (
            "```json\n"
            '[{"name":"Sarah","kind":"person","canonical_name":"sarah","confidence":0.9}]\n'
            "```"
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].name == "Sarah"

    def test_bare_fence(self):
        raw = (
            "```\n"
            '[{"name":"Sarah","kind":"person","canonical_name":"sarah","confidence":0.9}]\n'
            "```"
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1


class TestCoerceJsonTrailingProse:
    def test_prose_before_and_after(self):
        """LLMs sometimes add prose around the JSON payload."""
        raw = (
            "Here are the entities I found:\n"
            '[{"name":"Acme","kind":"organization",'
            '"canonical_name":"acme","confidence":0.8}]\n'
            "Let me know if you need more."
        )
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].name == "Acme"


class TestCoerceJsonMalformed:
    def test_empty_string(self):
        assert _coerce_json_to_entities("") == []

    def test_whitespace_only(self):
        assert _coerce_json_to_entities("   \n\t  ") == []

    def test_pure_prose(self):
        assert _coerce_json_to_entities("I couldn't find any entities.") == []

    def test_invalid_json(self):
        assert _coerce_json_to_entities("[{bad json}]") == []

    def test_non_list_non_dict_top_level(self):
        assert _coerce_json_to_entities('"just a string"') == []
        assert _coerce_json_to_entities("42") == []


class TestCoerceJsonFieldCoercion:
    def test_missing_name_dropped(self):
        raw = '[{"kind":"person","canonical_name":"x","confidence":0.9}]'
        assert _coerce_json_to_entities(raw) == []

    def test_empty_name_dropped(self):
        raw = '[{"name":"","kind":"person","canonical_name":"x","confidence":0.9}]'
        assert _coerce_json_to_entities(raw) == []

    def test_missing_kind_dropped(self):
        raw = '[{"name":"X","canonical_name":"x","confidence":0.9}]'
        assert _coerce_json_to_entities(raw) == []

    def test_invalid_kind_dropped(self):
        raw = '[{"name":"X","kind":"food","canonical_name":"x","confidence":0.9}]'
        assert _coerce_json_to_entities(raw) == []

    def test_missing_canonical_name_derived_from_name(self):
        raw = '[{"name":"Sarah Chen","kind":"person","confidence":0.9}]'
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].canonical_name == "sarah chen"

    def test_missing_confidence_defaults_to_0_5(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x"}]'
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].confidence == 0.5

    def test_confidence_clamped_above(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":1.5}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].confidence == 1.0

    def test_confidence_clamped_below(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":-0.5}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].confidence == 0.0

    def test_confidence_non_numeric_defaults(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":"high"}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].confidence == 0.5

    def test_kind_case_normalized(self):
        raw = '[{"name":"X","kind":"PERSON","canonical_name":"x","confidence":0.9}]'
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].kind == "person"

    def test_non_dict_items_skipped(self):
        raw = '["just a string", {"name":"X","kind":"person","canonical_name":"x","confidence":0.9}]'
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].name == "X"

    def test_aliases_non_list_becomes_empty(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":0.9,"aliases":"not a list"}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].aliases == []

    def test_aliases_list_normalized(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":0.9,"aliases":["A1","A2",""]}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].aliases == ["A1", "A2"]  # empty strings filtered

    def test_attributes_non_dict_becomes_empty(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":0.9,"attributes":"not a dict"}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].attributes == {}

    def test_attributes_dict_preserved(self):
        raw = '[{"name":"X","kind":"person","canonical_name":"x","confidence":0.9,"attributes":{"role":"CEO","city":"NYC"}}]'
        out = _coerce_json_to_entities(raw)
        assert out[0].attributes == {"role": "CEO", "city": "NYC"}


class TestCoerceJsonAllEntityKinds:
    """Verify all 5 entity kinds are accepted (parity with entities.VALID_KINDS)."""

    @pytest.mark.parametrize(
        "kind", ["person", "organization", "project", "location", "concept"]
    )
    def test_valid_kind_accepted(self, kind):
        raw = f'[{{"name":"X","kind":"{kind}","canonical_name":"x","confidence":0.9}}]'
        out = _coerce_json_to_entities(raw)
        assert len(out) == 1
        assert out[0].kind == kind


# ─── OllamaLLMExtractor ─────────────────────────────────────────────────


class TestOllamaExtractorConstruction:
    def test_defaults(self):
        e = OllamaLLMExtractor()
        assert e._host == DEFAULT_OLLAMA_HOST.rstrip("/")
        assert e._model == DEFAULT_EXTRACTION_MODEL

    def test_host_trailing_slash_stripped(self):
        e = OllamaLLMExtractor(host="http://localhost:11434/")
        assert e._host == "http://localhost:11434"

    def test_custom_model(self):
        e = OllamaLLMExtractor(model="llama3:8b")
        assert e._model == "llama3:8b"


class TestOllamaExtractorExtract:
    def test_empty_text_returns_empty(self):
        ext = _FakeOllamaExtractor(available=True)
        assert ext.extract("") == []
        # Should not invoke the LLM at all
        assert ext.call_count == 0

    def test_whitespace_only_text_returns_empty(self):
        ext = _FakeOllamaExtractor(available=True)
        assert ext.extract("   \n\t  ") == []
        assert ext.call_count == 0

    def test_unavailable_backend_returns_empty(self):
        ext = _FakeOllamaExtractor(available=False)
        assert ext.extract("Sarah met with Acme Corp last week.") == []
        # Unavailable means we never call _call_llm
        assert ext.call_count == 0

    def test_happy_path_returns_parsed_entities(self):
        script = [
            '[{"name":"Sarah Chen","kind":"person",'
            '"canonical_name":"sarah chen","confidence":0.9},'
            '{"name":"Acme Corp","kind":"organization",'
            '"canonical_name":"acme corp","confidence":0.8}]'
        ]
        ext = _FakeOllamaExtractor(script=script, available=True)
        result = ext.extract("Sarah met with Acme Corp last week.")
        assert len(result) == 2
        assert result[0].name == "Sarah Chen"
        assert result[0].kind == "person"
        assert result[1].kind == "organization"
        assert ext.call_count == 1

    def test_source_ref_propagates_to_all_entities(self):
        script = [
            '[{"name":"A","kind":"person","canonical_name":"a","confidence":0.9},'
            '{"name":"B","kind":"person","canonical_name":"b","confidence":0.9}]'
        ]
        ext = _FakeOllamaExtractor(script=script, available=True)
        result = ext.extract("text", source_ref="session-abc")
        assert all(e.source_ref == "session-abc" for e in result)

    def test_http_error_swallowed_returns_empty(self):
        script = [RuntimeError("boom")]
        ext = _FakeOllamaExtractor(script=script, available=True)
        result = ext.extract("Some text about Sarah.")
        assert result == []
        assert ext.call_count == 1

    def test_malformed_response_returns_empty(self):
        script = ["this is not JSON at all, just prose"]
        ext = _FakeOllamaExtractor(script=script, available=True)
        result = ext.extract("Text about things.")
        assert result == []
        assert ext.call_count == 1

    def test_empty_response_returns_empty(self):
        script = [""]
        ext = _FakeOllamaExtractor(script=script, available=True)
        assert ext.extract("text") == []

    def test_multiple_extract_calls_each_consume_script(self):
        script = [
            '[{"name":"A","kind":"person","canonical_name":"a","confidence":0.9}]',
            '[{"name":"B","kind":"person","canonical_name":"b","confidence":0.9}]',
        ]
        ext = _FakeOllamaExtractor(script=script, available=True)
        r1 = ext.extract("first")
        r2 = ext.extract("second")
        assert r1[0].name == "A"
        assert r2[0].name == "B"
        assert ext.call_count == 2


class TestOllamaExtractorAvailability:
    """Availability probe behavior.

    Probing calls ``_call_llm`` with a short probe prompt. Tests use
    a subclass override of ``_call_llm`` to script the probe result,
    mirroring how ``OllamaEmbedder`` tests stub ``_call_embed``.
    """

    def test_not_raise_on_connection_error(self):
        """A failed probe must not raise. Must return False silently."""

        class _DeadExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                raise ConnectionError("no route to host")

        ext = _DeadExtractor()
        assert ext.is_available() is False

    def test_cached_after_first_call(self):
        """After the first probe, subsequent calls should not re-probe."""

        probe_count = 0

        class _CountingExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                nonlocal probe_count
                probe_count += 1
                raise ConnectionError("fail")

        ext = _CountingExtractor()
        assert ext.is_available() is False
        assert ext.is_available() is False
        assert ext.is_available() is False
        assert probe_count == 1  # probe only runs once

    def test_successful_probe_returns_true(self):
        """Non-empty response sets availability to True."""

        class _OkExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                return "[]"  # valid JSON array response

        ext = _OkExtractor()
        assert ext.is_available() is True

    def test_empty_probe_response_returns_false(self):
        """An empty string response is treated as a degraded daemon."""

        class _EmptyExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                return ""

        ext = _EmptyExtractor()
        assert ext.is_available() is False

    def test_reset_availability_forces_reprobe(self):
        """reset_availability() clears the cache so next call re-probes."""

        results = ["first_ok", ConnectionError("fail_second")]

        class _FlippingExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                item = results.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        ext = _FlippingExtractor()
        assert ext.is_available() is True
        ext.reset_availability()
        assert ext.is_available() is False

    def test_no_probe_on_construction(self):
        """Invariant #1: initialize()/construction must not make network calls.

        The OllamaLLMExtractor constructor must NOT call _call_llm.
        """

        probe_count = 0

        class _TrackedExtractor(OllamaLLMExtractor):
            def _call_llm(self, prompt):  # type: ignore[override]
                nonlocal probe_count
                probe_count += 1
                return "[]"

        _TrackedExtractor()  # construction only
        assert probe_count == 0


# ─── Prompt invariants ──────────────────────────────────────────────────


class TestPromptInvariants:
    """The extraction prompt must instruct the LLM to skip certain categories.

    These are tripwire tests: if the prompt is edited and loses these
    guardrails, an LLM will start returning "today", "Monday", "the team",
    etc., as entities. The prompt is the ONLY place these filters exist
    in Phase 2B.1 (no post-hoc stopword list, unlike v1). Losing the
    guardrails is a regression even if the tests that mock the LLM
    still pass.
    """

    def test_prompt_mentions_temporal_filter(self):
        from plugins.memory.claudia.extractor import _EXTRACTION_PROMPT

        lower = _EXTRACTION_PROMPT.lower()
        # Explicitly rule out date-like tokens
        assert "date" in lower or "temporal" in lower or "days" in lower

    def test_prompt_mentions_common_noun_filter(self):
        from plugins.memory.claudia.extractor import _EXTRACTION_PROMPT

        lower = _EXTRACTION_PROMPT.lower()
        # Explicitly rule out common / generic nouns
        assert "common noun" in lower or "proper noun" in lower

    def test_prompt_mentions_empty_array_fallback(self):
        """The prompt must tell the LLM to return [] if no entities found."""
        from plugins.memory.claudia.extractor import _EXTRACTION_PROMPT

        assert "[]" in _EXTRACTION_PROMPT

    def test_prompt_mentions_all_entity_kinds(self):
        """All 5 kinds must appear in the prompt so the LLM knows the taxonomy."""
        from plugins.memory.claudia.extractor import _EXTRACTION_PROMPT

        lower = _EXTRACTION_PROMPT.lower()
        for kind in ["person", "organization", "project", "location", "concept"]:
            assert kind in lower, f"prompt missing kind: {kind}"

    def test_prompt_includes_input_text(self):
        """The prompt format string must have a {text} placeholder."""
        from plugins.memory.claudia.extractor import _EXTRACTION_PROMPT

        assert "{text}" in _EXTRACTION_PROMPT


# ─── LLMExtractor ABC ───────────────────────────────────────────────────


class TestLLMExtractorABC:
    def test_cannot_instantiate_abc(self):
        """Calling LLMExtractor() directly raises because extract is abstract."""
        with pytest.raises(TypeError):
            LLMExtractor()  # type: ignore[abstract]

    def test_default_is_available_is_true(self):
        """Concrete subclasses get is_available=True unless they override."""

        class _Minimal(LLMExtractor):
            def extract(self, text, *, source_ref=""):
                return []

        assert _Minimal().is_available() is True
