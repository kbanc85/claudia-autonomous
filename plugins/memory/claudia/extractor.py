"""LLM-based entity extraction for Claudia memory (Phase 2B.1).

Replaces v1's spaCy + regex ``EntityExtractor`` with an LLM-prompted
extractor that asks a small local model to identify named entities
in conversation text and return them as a structured JSON array.

Design principles:

- **Lazy availability probing (Invariant #1).** ``initialize()``
  does not touch the network. The first ``is_available()`` or
  ``extract()`` call probes the Ollama daemon; the result is
  cached under a lock and reused.

- **Prompt over post-hoc filtering.** v1 maintained a ~100-entry
  ``STOP_WORDS`` list to catch spaCy false-positives like "drawn",
  "overall", "recently". An LLM instructed to extract only proper
  nouns and to skip dates/pronouns does this natively, so Phase
  2B.1 ships no post-hoc stopword list. If the LLM drifts, add
  guidance to the prompt — do not reintroduce the stopword list.

- **Best-effort extraction.** ``extract()`` never raises. HTTP
  errors, JSON parse errors, missing fields, invalid kinds —
  everything reduces to an empty list. The provider's write path
  continues normally even when extraction fails, because the
  conversation turn itself is preserved in ``memories``.

- **Fire-and-forget threading (Phase 2B.1 wire-up).** The provider
  runs ``extract()`` on a dedicated single-worker
  ``ThreadPoolExecutor`` — NOT on the caller's thread and NOT on
  the writer thread. This diverges from the embedding path (caller
  thread) because LLM calls are 10-100x slower than embeddings and
  would violate Invariant #3 (``sync_turn`` must be non-blocking)
  if inlined. It diverges from the writer thread because extraction
  is too slow to gate writes; subsequent turns' writes would stall
  behind the first turn's extraction.

- **Testability via ``_call_llm`` override.** Tests subclass
  ``OllamaLLMExtractor`` and override ``_call_llm`` to return
  scripted responses. No httpx monkeypatching required. This
  mirrors the ``OllamaEmbedder._call_embed`` pattern from Phase
  2A.2b.

- **Tolerant JSON parsing.** ``_coerce_json_to_entities`` handles
  plain arrays, object-wrapped arrays, fenced markdown, and prose
  with embedded JSON. Missing fields default to sane values;
  invalid fields are dropped silently. The contract is "garbage in,
  empty list out".

Public API:

- ``ExtractedEntity`` — dataclass mirroring the entities table
  columns: ``name``, ``kind``, ``canonical_name``, ``confidence``,
  ``aliases``, ``attributes``, ``source_ref``
- ``LLMExtractor`` — ABC. ``extract(text, *, source_ref)`` returns
  a list; ``is_available()`` returns a bool
- ``OllamaLLMExtractor`` — concrete implementation backed by a
  local Ollama daemon (default host ``http://localhost:11434``,
  default model ``qwen2.5:3b``). Env var ``CLAUDIA_OLLAMA_HOST``
  overrides the host for shared setups

Reference: docs/decisions/memory-provider-design.md (Phase 2B.1
extraction section) and autonomous/fork/plans/phase-2b-handoff.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Defaults ────────────────────────────────────────────────────────────

#: Default Ollama host URL. Override via ``CLAUDIA_OLLAMA_HOST``.
#: Shared with OllamaEmbedder so a single env var points both at the
#: same daemon in typical setups.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

#: Environment variable that overrides the default host.
ENV_HOST = "CLAUDIA_OLLAMA_HOST"

#: Default extraction model. Small, fast, good enough for entity
#: identification. Users can point at a larger model via config in
#: a later sub-task.
DEFAULT_EXTRACTION_MODEL = "qwen2.5:3b"

#: Default HTTP timeout in seconds. Generous enough for a local
#: ~3B model, tight enough that a stuck daemon does not stall
#: extraction queue processing forever.
DEFAULT_TIMEOUT_S = 30.0

#: The valid entity kinds, matching ``entities.VALID_KINDS``. If
#: these drift apart ``upsert_entity`` will reject the extractor's
#: output with ``ValueError: invalid kind``. A tripwire test locks
#: the relationship.
VALID_ENTITY_KINDS = frozenset({
    "person",
    "organization",
    "project",
    "location",
    "concept",
})


# ─── Public types ────────────────────────────────────────────────────────


@dataclass
class ExtractedEntity:
    """A single entity extracted from conversation text.

    Shape matches the entities table via ``upsert_entity``:

    - ``name``: display form, stored as ``entities.name``
    - ``kind``: one of VALID_ENTITY_KINDS
    - ``canonical_name``: lowercase normalized form, used for
      human-inspection / debugging (the DB derives its own
      canonical form via LOWER() in queries)
    - ``confidence``: 0.0..1.0, passed to ``upsert_entity`` as
      the initial ``importance`` for a new row. Existing rows are
      NOT overwritten by a lower confidence (Phase 2B.3 will add
      proper importance accumulation)
    - ``aliases``: additional surface forms the LLM identified
      in the text
    - ``attributes``: structured side-data (role, city, industry,
      etc.). Opaque to the extractor; the provider stores them as
      ``attributes_json``
    - ``source_ref``: the session_id or source-ref of the turn
      this entity came from, for provenance
    """

    name: str
    kind: str
    canonical_name: str
    confidence: float
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_ref: str = ""


# ─── Abstract base class ─────────────────────────────────────────────────


class LLMExtractor(ABC):
    """Abstract base class for entity extractors.

    Implementations:

    - ``OllamaLLMExtractor`` in this module for real extraction
    - tests inject a ``LLMExtractor`` subclass that returns scripted
      results without touching the network

    The contract is intentionally minimal so new backends (remote
    OpenAI, Anthropic, llama.cpp, etc.) can slot in without changes
    to the provider wire-up.
    """

    @abstractmethod
    def extract(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[ExtractedEntity]:
        """Extract entities from ``text``.

        Must never raise. Implementations should log and swallow
        errors, returning an empty list for:

        - Empty / whitespace text
        - Unavailable backend
        - HTTP / parse / network errors
        - Malformed LLM responses
        - No detected entities

        ``source_ref`` should be stamped onto each returned entity's
        ``source_ref`` field for provenance.
        """

    def is_available(self) -> bool:
        """Return True if the extractor can make calls right now.

        Default returns True. Network-backed subclasses should
        override to probe lazily (after the first call, NOT in
        ``__init__``).
        """
        return True


# ─── Ollama implementation ───────────────────────────────────────────────


#: The extraction prompt. The LLM is instructed to return a JSON
#: array only. The guardrails (no common nouns, no temporal tokens,
#: no invented entities) are the ONLY place these filters exist in
#: Phase 2B.1 — the post-hoc stopword list from v1 is gone. Tripwire
#: tests (test_extractor.py::TestPromptInvariants) verify this prompt
#: contains the expected keywords; update both the prompt AND the
#: tripwires together if you edit this.
_EXTRACTION_PROMPT = """You are an entity extractor. Identify named entities in the text and return them as a JSON array.

For each entity, return an object with these fields:
- "name": the entity as it appears in the text
- "kind": one of "person", "organization", "project", "location", "concept"
- "canonical_name": lowercase normalized form
- "confidence": 0.0 to 1.0
- "aliases": list of alternative names mentioned for this entity (empty if none)

Rules:
- Only extract proper nouns (specific people, companies, projects, places, named concepts).
- Do not extract common nouns, adjectives, verbs, or generic terms.
- Do not extract days of the week, months, dates, or temporal expressions.
- Do not extract pronouns or possessives ("the team", "our company", "my friend").
- Do not invent entities not present in the text.
- If the text has no named entities, return an empty array [].

Return ONLY the JSON array, no prose.

Text:
{text}

JSON:"""


#: Short probe prompt. Small and neutral. Used by is_available() to
#: verify the daemon + model are live without loading a full turn.
_PROBE_PROMPT = "Return exactly this JSON: []"


class OllamaLLMExtractor(LLMExtractor):
    """LLM extractor backed by a local Ollama daemon.

    Lifecycle mirrors ``OllamaEmbedder``:

        extractor = OllamaLLMExtractor()        # no network
        # ... later, from the extraction worker ...
        if extractor.is_available():            # lazy probe
            entities = extractor.extract(text)  # real call
        # on shutdown: nothing to close (short-lived httpx clients)

    Subclasses override ``_call_llm`` for testability. Do not
    override ``extract`` or ``is_available`` — those contain the
    cache/locking logic that downstream code relies on.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        model: str = DEFAULT_EXTRACTION_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._host = (
            host or os.environ.get(ENV_HOST) or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
        self._model = model
        self._timeout = timeout

        # _available: None = unprobed, True = known available,
        #             False = known unavailable (no re-probe without reset)
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def host(self) -> str:
        return self._host

    @property
    def model(self) -> str:
        return self._model

    # ── Availability ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Lazy, cached probe. First call runs ``_probe()``."""
        with self._lock:
            if self._available is not None:
                return self._available

        # Release the lock for the network round-trip so other
        # threads aren't blocked. Multiple concurrent probes are
        # harmless — last-writer-wins, all writers see the same result.
        probe_ok = self._probe()

        with self._lock:
            self._available = probe_ok
            return self._available

    def reset_availability(self) -> None:
        """Clear the cached availability so the next call re-probes.

        Useful after a session boundary, after starting the Ollama
        daemon manually, or in tests.
        """
        with self._lock:
            self._available = None

    def _probe(self) -> bool:
        """Run a single tiny LLM call to verify the daemon + model work."""
        try:
            raw = self._call_llm(_PROBE_PROMPT)
        except Exception as exc:
            logger.debug("Ollama extractor probe failed: %s", exc)
            return False
        return bool(raw)

    # ── Extraction ────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[ExtractedEntity]:
        """Extract entities via the LLM. Returns [] on any failure."""
        if not text or not text.strip():
            return []
        if not self.is_available():
            return []

        prompt = _EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            logger.warning("Ollama extraction call failed: %s", exc)
            return []

        if not raw:
            return []

        return _coerce_json_to_entities(raw, source_ref=source_ref)

    # ── Overridable for tests ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """POST a prompt to Ollama's ``/api/generate`` endpoint.

        Short-lived ``httpx.Client`` per call (matches ``OllamaEmbedder``
        pattern: no persistent client to manage across the extractor's
        lifetime). ``format: "json"`` asks Ollama to constrain its
        output to valid JSON, which dramatically improves parse
        reliability for smaller models.

        Returns the raw ``response`` field from Ollama's reply. On
        non-200 status, returns an empty string so downstream code
        treats it as "no extraction".
        """
        import httpx  # noqa: PLC0415 - lazy import matches OllamaEmbedder

        url = f"{self._host}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)

        if response.status_code != 200:
            logger.debug(
                "Ollama /api/generate returned status %d: %s",
                response.status_code,
                response.text[:200],
            )
            return ""

        try:
            data = response.json()
        except ValueError:
            logger.debug("Ollama /api/generate returned non-JSON body")
            return ""

        if not isinstance(data, dict):
            return ""

        raw = data.get("response", "")
        return raw if isinstance(raw, str) else ""


# ─── JSON coercion ───────────────────────────────────────────────────────


#: Pattern to extract the first top-level JSON array or object from
#: a string that may have prose around it. Greedy inside brackets so
#: we don't truncate nested objects; we only need the OUTER pair.
_JSON_SNIFF_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)

#: Keys used to unwrap an object-shaped LLM response. LLMs sometimes
#: return ``{"entities": [...]}`` instead of a bare array even when
#: the prompt asks for an array. Accept common synonyms.
_OBJECT_UNWRAP_KEYS = ("entities", "results", "items", "result")


def _coerce_json_to_entities(
    raw: str,
    *,
    source_ref: str = "",
) -> List[ExtractedEntity]:
    """Tolerant parser from LLM output text to ``ExtractedEntity`` list.

    Handles:

    - Plain JSON array ``[{...}, {...}]``
    - Object-wrapped arrays ``{"entities": [...]}``,
      ``{"results": [...]}``, etc.
    - Fenced markdown code blocks ``` ```json\\n...\\n``` ```
    - Prose with embedded JSON (finds the first ``[...]`` or ``{...}``)
    - Missing fields (defaults: canonical_name from name, confidence=0.5,
      aliases=[], attributes={})
    - Out-of-range confidence (clamped to [0, 1])
    - Non-numeric confidence (defaults to 0.5)
    - Invalid ``kind`` (entity dropped)
    - Missing or empty ``name`` (entity dropped)
    - Non-dict items inside the array (skipped)

    Returns an empty list for any unrecoverable input. Never raises.
    """
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        if not raw:
            return []

    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to locate a top-level JSON structure inside prose
        match = _JSON_SNIFF_RE.search(raw)
        if match is None:
            return []
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

    # Unwrap {"entities": [...]} forms
    if isinstance(parsed, dict):
        unwrapped: Any = None
        for key in _OBJECT_UNWRAP_KEYS:
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                unwrapped = candidate
                break
        if unwrapped is None:
            return []
        parsed = unwrapped

    if not isinstance(parsed, list):
        return []

    entities: List[ExtractedEntity] = []
    for item in parsed:
        ent = _coerce_one_entity(item, source_ref=source_ref)
        if ent is not None:
            entities.append(ent)
    return entities


def _coerce_one_entity(
    item: Any,
    *,
    source_ref: str,
) -> Optional[ExtractedEntity]:
    """Coerce a single JSON-decoded dict into an ExtractedEntity.

    Returns None if the item is unusable (not a dict, missing
    required fields, invalid kind).
    """
    if not isinstance(item, dict):
        return None

    # Name: required, non-empty after strip
    name_raw = item.get("name", "")
    if not isinstance(name_raw, str):
        return None
    name = name_raw.strip()
    if not name:
        return None

    # Kind: required, normalized to lowercase, must be in VALID_ENTITY_KINDS
    kind_raw = item.get("kind", "")
    if not isinstance(kind_raw, str):
        return None
    kind = kind_raw.strip().lower()
    if kind not in VALID_ENTITY_KINDS:
        return None

    # Canonical name: optional, defaults to lower(name)
    canonical_raw = item.get("canonical_name", "")
    if isinstance(canonical_raw, str):
        canonical_name = canonical_raw.strip().lower()
    else:
        canonical_name = ""
    if not canonical_name:
        canonical_name = name.lower()

    # Confidence: optional, clamped to [0, 1], defaults to 0.5
    confidence = _coerce_confidence(item.get("confidence"))

    # Aliases: optional list of non-empty strings
    aliases = _coerce_aliases(item.get("aliases"))

    # Attributes: optional dict
    attributes = _coerce_attributes(item.get("attributes"))

    return ExtractedEntity(
        name=name,
        kind=kind,
        canonical_name=canonical_name,
        confidence=confidence,
        aliases=aliases,
        attributes=attributes,
        source_ref=source_ref,
    )


def _coerce_confidence(value: Any) -> float:
    """Clamp to [0.0, 1.0]. Defaults to 0.5 on anything non-numeric."""
    if value is None:
        return 0.5
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _coerce_aliases(value: Any) -> List[str]:
    """Normalize to a list of non-empty strings. Non-list → []."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def _coerce_attributes(value: Any) -> Dict[str, Any]:
    """Pass through dicts, reject everything else."""
    if isinstance(value, dict):
        return dict(value)
    return {}
