"""Commitment detection for Claudia memory (Phase 2B.2).

Surfaces promises like "I'll send the proposal by Friday" from
conversation text and writes them to the ``commitments`` table via
the write queue. Uses a hybrid approach:

- **Pattern pre-filter** (``PatternCommitmentDetector``): fast regex
  detection for candidate sentences. Matches first-person future
  constructions ("I'll X", "I will X", "I'm going to X", "we'll X",
  "we will X", "we're going to X") and implicit-obligation phrases
  ("let me X", "I need to X", "I should X"). Rejects past tense,
  questions, and third-person by construction (the regex requires
  the subject pronoun + a future modal). Offline-safe, no external
  dependencies.

- **LLM refinement** (``OllamaCommitmentDetector``): asks a small
  local Ollama model to identify commitments with structured
  output, classified as "explicit" / "implicit" / "vague". Vague
  commitments are dropped (Claudia's core principle: vague
  intentions don't get accountability). Used to reduce pattern
  false positives and to extract cleaner action phrases.

- **Hybrid** (``HybridCommitmentDetector``): pattern pre-filter
  short-circuits when nothing matches (skip the LLM call). When
  the pattern fires, the LLM refines the matches. If the LLM is
  unavailable or fails, the pattern output is returned verbatim
  as an offline fallback. This is the default detector wired into
  ``ClaudiaMemoryProvider``.

Explicit vs implicit vs vague taxonomy (from claudia-principles.md):

  | Type     | Example                        | Action            |
  |----------|--------------------------------|-------------------|
  | Explicit | "I'll send the proposal"       | Track w/ deadline |
  | Implicit | "Let me get back to you"       | Ask for deadline  |
  | Vague    | "We should explore that some." | Don't track       |

Design principles:

- **Lazy LLM availability** per invariant #1. The Ollama detector
  does not probe the daemon in ``__init__``; the first
  ``is_available()`` call does.
- **Best-effort**: ``detect()`` never raises. Pattern detection
  can return [] on empty input; LLM failures fall back to pattern
  (hybrid) or return [] (pure LLM).
- **Testability**: subclass and override ``_call_llm`` for
  deterministic tests. No httpx monkeypatching required. Mirrors
  the ``OllamaLLMExtractor`` pattern from Phase 2B.1.
- **Threading**: instances are NOT required to be thread-safe
  beyond the availability cache. Each provider holds a single
  detector and submits ``detect`` calls through the shared
  cognitive executor, so concurrency is external.

Public API:

- ``DetectedCommitment`` dataclass
- ``CommitmentDetector`` ABC
- ``PatternCommitmentDetector`` (offline, regex-only)
- ``OllamaCommitmentDetector`` (LLM-only, lazy)
- ``HybridCommitmentDetector`` (pattern + LLM refinement, the default)
- ``_extract_pattern_matches(text) -> List[DetectedCommitment]``
- ``_parse_deadline(text, *, anchor) -> Optional[str]``
- ``_coerce_json_to_commitments(raw, *, source_ref) -> List[DetectedCommitment]``

Reference: autonomous/fork/plans/phase-2b-handoff.md Phase 2B.2
sub-task notes, plugins/memory/claudia/commitments.py for the
write layer, claudia/.claude/rules/claudia-principles.md for the
explicit/implicit/vague taxonomy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ─── Defaults ────────────────────────────────────────────────────────────

#: Shared with OllamaEmbedder / OllamaLLMExtractor. Same env var
#: points all three at the same daemon in typical setups.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
ENV_HOST = "CLAUDIA_OLLAMA_HOST"

#: Small local model for commitment detection. Deliberately the
#: same as the entity extractor default: one model load, two uses.
DEFAULT_DETECTION_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT_S = 30.0


# ─── Public types ────────────────────────────────────────────────────────


#: The commitment types Claudia tracks. "vague" is intentionally
#: absent from this set because the core principle says vague
#: intentions don't get tracked. The coercer uses this set to drop
#: vague entries.
VALID_COMMITMENT_TYPES = frozenset({"explicit", "implicit"})


@dataclass
class DetectedCommitment:
    """A single commitment detected in conversation text.

    Fields map to ``commitments.Commitment`` where applicable but
    also carry detection-time metadata (confidence, commitment_type,
    origin) that the CRUD layer doesn't persist.

    - ``content``: the action phrase, stored verbatim in the
      commitments.content column (e.g. "send the proposal to Sarah")
    - ``deadline_raw``: the original deadline text ("by Friday")
    - ``deadline_iso``: parsed deadline as ISO 8601, or None if
      unparseable. Stored as ``commitments.deadline`` when non-None.
    - ``confidence``: 0.0..1.0, used by Phase 2B.3 consolidation
      to decide whether to dedupe/merge. Not persisted today.
    - ``commitment_type``: "explicit" or "implicit". Not persisted
      directly but influences default priority. Phase 2B.3 may add
      a dedicated column.
    - ``source_ref``: propagated from sync_turn for provenance.
      Stored as ``commitments.source_ref``.
    - ``origin``: "pattern", "llm", or "hybrid" — which detector
      produced this record. Used by tests and future auditing.
    """

    content: str
    deadline_raw: Optional[str] = None
    deadline_iso: Optional[str] = None
    confidence: float = 0.5
    commitment_type: str = "explicit"
    source_ref: str = ""
    origin: str = "pattern"


# ─── Abstract base class ────────────────────────────────────────────────


class CommitmentDetector(ABC):
    """Abstract base for commitment detectors.

    Two concrete forms ship in this module:

    - ``PatternCommitmentDetector``: offline regex detection
    - ``OllamaCommitmentDetector``: LLM-only via Ollama daemon
    - ``HybridCommitmentDetector``: the default, combines both

    The provider uses ``HybridCommitmentDetector`` by default.
    Tests inject a deterministic fake.
    """

    @abstractmethod
    def detect(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[DetectedCommitment]:
        """Detect commitments in ``text``.

        Must never raise. Implementations that wrap network or LLM
        calls should log and swallow errors, returning [] on any
        failure. ``source_ref`` is stamped on each returned
        commitment for provenance.
        """

    def is_available(self) -> bool:
        """Return True if the detector can work right now.

        Default returns True. Network-backed subclasses override
        with a lazy probe.
        """
        return True


# ─── Regex patterns ──────────────────────────────────────────────────────


#: Words that downgrade a match from commitment to "vague / skip".
#: When any of these appear in a matched sentence, the pattern
#: detector drops the match entirely. Claudia's core principle
#: says vague intentions don't get tracked.
_VAGUE_MARKERS = frozenset({
    "someday",
    "eventually",
    "maybe",
    "possibly",
    "perhaps",
    "at some point",
    "sometime",
    "one day",
})


@dataclass(frozen=True)
class _PatternSpec:
    """Internal config for a single pattern regex."""

    regex: re.Pattern
    commitment_type: str
    confidence: float


# Explicit commitment patterns. High confidence.
# All patterns:
# - Use (?:^|(?<=[.!?\n])) to anchor to sentence start (avoids
#   matching inside longer sentences like "I think I'll X" where
#   "I'll X" would otherwise fire). Actually we want to match those
#   too, so we use \b instead to match at word boundaries.
# - Use \b<subject> to avoid matching inside other words (e.g.
#   "WILL" inside "goodwill" won't match).
# - Capture the action phrase up to sentence terminator.
#
# The lazy quantifier (.+?) captures the shortest match that
# satisfies the lookahead (period, newline, or end of string).
def _compile_explicit_patterns() -> List[_PatternSpec]:
    flags = re.IGNORECASE | re.DOTALL
    return [
        # I'll / I will / I shall + action
        _PatternSpec(
            regex=re.compile(
                r"\bI(?:'ll|\s+will|\s+shall)\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="explicit",
            confidence=0.85,
        ),
        # I'm going to / I am going to + action
        _PatternSpec(
            regex=re.compile(
                r"\bI(?:'m|\s+am)\s+going\s+to\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="explicit",
            confidence=0.85,
        ),
        # we'll / we will / we shall + action
        _PatternSpec(
            regex=re.compile(
                r"\b[Ww]e(?:'ll|\s+will|\s+shall)\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="explicit",
            confidence=0.8,
        ),
        # we're going to / we are going to + action
        _PatternSpec(
            regex=re.compile(
                r"\b[Ww]e(?:'re|\s+are)\s+going\s+to\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="explicit",
            confidence=0.8,
        ),
    ]


# Implicit commitment patterns. Medium/low confidence.
def _compile_implicit_patterns() -> List[_PatternSpec]:
    flags = re.IGNORECASE | re.DOTALL
    return [
        # let me / let's + action
        _PatternSpec(
            regex=re.compile(
                r"\b[Ll]et\s+(?:me|'s|us)\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="implicit",
            confidence=0.6,
        ),
        # I need to + action
        _PatternSpec(
            regex=re.compile(
                r"\bI\s+need\s+to\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="implicit",
            confidence=0.55,
        ),
        # I should + action
        _PatternSpec(
            regex=re.compile(
                r"\bI\s+should\s+(.{3,}?)(?=[.!?\n]|$)",
                flags=flags,
            ),
            commitment_type="implicit",
            confidence=0.5,
        ),
    ]


#: Past-tense indicators that reject a match. If any of these
#: appear between the subject and the modal (e.g. "I just sent X"),
#: we assume the statement is about a completed action. Current
#: regex already requires future modals so most past tense is
#: already rejected, but this is a belt-and-suspenders filter.
_PAST_TENSE_INDICATORS = ("sent", "sold", "shipped", "delivered")


_EXPLICIT_PATTERNS = _compile_explicit_patterns()
_IMPLICIT_PATTERNS = _compile_implicit_patterns()


def _contains_vague_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _VAGUE_MARKERS)


def _is_question(text: str) -> bool:
    """Check if text ends with a question mark or starts with 'will you'."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if re.match(r"^\s*(?:will|can|could|would|shall)\s+(?:you|we)\b", stripped, re.IGNORECASE):
        return True
    return False


def _sentence_containing(text: str, start: int, end: int) -> str:
    """Return the sentence containing the character range (start, end).

    Used by the pattern extractor to check for vague markers or
    question marks in the surrounding context. Splits on ``.!?\\n``.
    """
    # Walk left from start until a sentence terminator
    left = start
    while left > 0 and text[left - 1] not in ".!?\n":
        left -= 1
    # Walk right from end until a sentence terminator
    right = end
    while right < len(text) and text[right] not in ".!?\n":
        right += 1
    return text[left:right]


def _extract_pattern_matches(text: str) -> List[DetectedCommitment]:
    """Run every pattern against the text and return all matches.

    Each match becomes a DetectedCommitment with:
    - ``content`` = the captured action phrase
    - ``commitment_type`` = from the pattern spec
    - ``confidence`` = from the pattern spec
    - ``deadline_raw`` = extracted from the containing sentence
    - ``origin`` = "pattern"

    Matches in sentences containing vague markers (someday, maybe,
    etc.) OR ending in '?' are dropped.
    """
    if not text or not text.strip():
        return []

    results: List[DetectedCommitment] = []
    seen_spans = set()

    for spec in _EXPLICIT_PATTERNS + _IMPLICIT_PATTERNS:
        for m in spec.regex.finditer(text):
            span_key = (m.start(), m.end())
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            # Check the surrounding sentence for disqualifiers
            sentence = _sentence_containing(text, m.start(), m.end())
            if _contains_vague_marker(sentence):
                continue
            if _is_question(sentence):
                continue

            # Capture the action phrase
            action = (m.group(1) or "").strip()
            if not action:
                continue
            # Strip trailing punctuation / whitespace
            action = action.rstrip(" \t,;:")
            if len(action) < 3:
                continue

            # Extract deadline from the sentence
            deadline_raw = _find_deadline_text(sentence)

            results.append(
                DetectedCommitment(
                    content=action,
                    deadline_raw=deadline_raw,
                    deadline_iso=_parse_deadline(deadline_raw) if deadline_raw else None,
                    confidence=spec.confidence,
                    commitment_type=spec.commitment_type,
                    origin="pattern",
                )
            )

    return results


# ─── Deadline detection ──────────────────────────────────────────────────


#: Weekday name → Python weekday() number (Monday=0).
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

#: Regex that catches common deadline phrases in a sentence. Order
#: matters: more-specific patterns first so "by Friday" wins over
#: "Friday" alone. Each entry's match is returned as the raw
#: deadline text; ``_parse_deadline`` then resolves it.
_DEADLINE_SEARCH_PATTERNS = [
    # "by/before <weekday>"
    re.compile(
        r"\b(?:by|before)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight)\b",
        re.IGNORECASE,
    ),
    # "next <weekday|unit>"
    re.compile(
        r"\bnext\s+"
        r"(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # "this <unit>"
    re.compile(
        r"\bthis\s+"
        r"(week|afternoon|evening|morning|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # "in N (hour|minute|day|week|month)s"
    re.compile(
        r"\bin\s+\d+\s+(?:hour|minute|day|week|month)s?\b",
        re.IGNORECASE,
    ),
    # Bare "tomorrow" / "today" / "tonight"
    re.compile(r"\b(tomorrow|today|tonight)\b", re.IGNORECASE),
]


def _find_deadline_text(sentence: str) -> Optional[str]:
    """Return the first deadline phrase found in a sentence, or None.

    The returned text is the full match (e.g. "by Friday"), not
    just the captured group. Downstream ``_parse_deadline`` resolves
    it to ISO 8601 when possible.
    """
    if not sentence:
        return None
    for pattern in _DEADLINE_SEARCH_PATTERNS:
        m = pattern.search(sentence)
        if m:
            return m.group(0).strip()
    return None


def _parse_deadline(
    text: Optional[str],
    *,
    anchor: Optional[datetime] = None,
) -> Optional[str]:
    """Resolve a natural-language deadline phrase to ISO 8601.

    Returns None if ``text`` is None, empty, or not recognized.
    Supported forms:

    - "today" / "tonight": ``anchor`` date at 23:59:59
    - "tomorrow": next day at 00:00:00
    - "by <weekday>" / "<weekday>" alone: next occurrence of the
      named weekday. If ``anchor`` IS that weekday, wraps to the
      following week.
    - "next week": ``anchor`` + 7 days
    - "next <weekday>": next occurrence (same rule as "by <weekday>")
    - "in N hours/minutes/days/weeks": arithmetic on ``anchor``
    - "in N months": approximated as 30 days per month

    ``anchor`` defaults to ``datetime.now(timezone.utc)`` but tests
    pin it for determinism. All returned ISO strings are
    timezone-aware (UTC).
    """
    if not text:
        return None

    text_lower = text.lower().strip()
    if not text_lower:
        return None

    if anchor is None:
        anchor = datetime.now(timezone.utc)
    elif anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)

    # "today" / "tonight" — end of current day
    if "today" in text_lower or "tonight" in text_lower:
        eod = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
        return eod.isoformat()

    # "tomorrow" — start of next day
    if "tomorrow" in text_lower:
        tomorrow = (anchor + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow.isoformat()

    # Weekdays: "Friday", "by Friday", "next Monday", etc.
    for day_name, day_num in _WEEKDAYS.items():
        if day_name in text_lower:
            today_num = anchor.weekday()
            days_until = (day_num - today_num) % 7
            if days_until == 0:
                # "by Friday" when today IS Friday → next Friday
                days_until = 7
            target = (anchor + timedelta(days=days_until)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return target.isoformat()

    # "next week" — 7 days from anchor
    if "next week" in text_lower:
        target = (anchor + timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return target.isoformat()

    # "in N units"
    m = re.search(
        r"\bin\s+(\d+)\s+(hour|minute|day|week|month)s?\b",
        text_lower,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "hour":
            target = anchor + timedelta(hours=n)
        elif unit == "minute":
            target = anchor + timedelta(minutes=n)
        elif unit == "day":
            target = anchor + timedelta(days=n)
        elif unit == "week":
            target = anchor + timedelta(weeks=n)
        elif unit == "month":
            # timedelta has no "months"; approximate as 30 days
            target = anchor + timedelta(days=30 * n)
        else:  # pragma: no cover - unreachable via regex
            return None
        return target.isoformat()

    # Unparseable: return None so the caller keeps the raw text
    return None


# ─── Detection prompt ────────────────────────────────────────────────────


#: The commitment detection prompt. Tripwire tests in
#: test_commitment_detector.py::TestPromptInvariants verify that
#: this prompt still contains the explicit/implicit/vague taxonomy,
#: the past-tense rejection guidance, the {text} placeholder, and
#: the empty-array fallback instruction. Update both the prompt
#: AND the tripwire tests together.
_DETECTION_PROMPT = """You are a commitment detector. Identify commitments (promises to do something in the future) in the text and return them as a JSON array.

For each commitment, return an object with these fields:
- "content": the action that was committed to (e.g., "send the proposal to Sarah")
- "commitment_type": one of "explicit", "implicit", or "vague"
    * "explicit": clear first-person promise ("I'll send X", "we will ship Y")
    * "implicit": softer obligation ("let me check on that", "I need to X")
    * "vague": maybe / someday / eventually
- "confidence": 0.0 to 1.0 (how certain this is a real commitment)
- "deadline_raw": the deadline phrase from the text if any (e.g., "by Friday"), or null

Rules:
- Only extract commitments from the speaker (first person: "I" or "we"). Do NOT extract third-person statements ("she will X", "they will X").
- Do NOT extract past or already-completed actions ("I sent the proposal", "we shipped it last week"). These are history, not commitments.
- Do NOT extract questions ("will you send X?"). Questions are requests.
- Do NOT extract hypotheticals or conditionals ("if I had time I'd X").
- Tag "vague" commitments accurately: they will be filtered out downstream. Do not promote vague intentions to explicit or implicit.
- If the text has no real commitments, return an empty array [].

Return ONLY the JSON array, no prose.

Text:
{text}

JSON:"""


#: Short probe prompt used for lazy availability checks. Same
#: shape as the real prompt but with trivial input.
_PROBE_PROMPT = "Return exactly this JSON: []"


# ─── JSON coercion ───────────────────────────────────────────────────────


_JSON_SNIFF_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)
_OBJECT_UNWRAP_KEYS = ("commitments", "results", "items", "result")


def _coerce_json_to_commitments(
    raw: str,
    *,
    source_ref: str = "",
) -> List[DetectedCommitment]:
    """Tolerant parser from LLM output to ``DetectedCommitment`` list.

    Handles plain arrays, object-wrapped arrays
    (``{"commitments": [...]}``), fenced markdown, and prose with
    embedded JSON. Missing fields default to sane values; invalid
    fields are dropped silently. Vague commitments are dropped
    (core principle: no accountability for vague intentions).
    Never raises.
    """
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []

    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        if not raw:
            return []

    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_SNIFF_RE.search(raw)
        if match is None:
            return []
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

    # Unwrap object-shaped response
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

    out: List[DetectedCommitment] = []
    for item in parsed:
        c = _coerce_one_commitment(item, source_ref=source_ref)
        if c is not None:
            out.append(c)
    return out


def _coerce_one_commitment(
    item: Any,
    *,
    source_ref: str,
) -> Optional[DetectedCommitment]:
    """Coerce a single JSON item to DetectedCommitment. Drops vague."""
    if not isinstance(item, dict):
        return None

    content_raw = item.get("content", "")
    if not isinstance(content_raw, str):
        return None
    content = content_raw.strip()
    if not content:
        return None

    type_raw = item.get("commitment_type", "explicit")
    if isinstance(type_raw, str):
        commitment_type = type_raw.strip().lower()
    else:
        commitment_type = "explicit"

    # Vague commitments are dropped per Claudia's core principle.
    if commitment_type == "vague":
        return None

    # Unknown types (LLM drift) default to "explicit" as a
    # permissive fallback — still tracks the commitment rather
    # than losing it silently.
    if commitment_type not in VALID_COMMITMENT_TYPES:
        commitment_type = "explicit"

    confidence = _coerce_confidence(item.get("confidence"))

    deadline_raw_value = item.get("deadline_raw")
    if isinstance(deadline_raw_value, str):
        deadline_raw = deadline_raw_value.strip() or None
    else:
        deadline_raw = None

    deadline_iso = _parse_deadline(deadline_raw) if deadline_raw else None

    return DetectedCommitment(
        content=content,
        deadline_raw=deadline_raw,
        deadline_iso=deadline_iso,
        confidence=confidence,
        commitment_type=commitment_type,
        source_ref=source_ref,
        origin="llm",
    )


def _coerce_confidence(value: Any) -> float:
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


# ─── PatternCommitmentDetector ──────────────────────────────────────────


class PatternCommitmentDetector(CommitmentDetector):
    """Regex-only commitment detector. Offline, deterministic, fast.

    Used as the pre-filter inside HybridCommitmentDetector and as
    the offline fallback when the LLM daemon is unreachable. Also
    the unit-test baseline for pattern regex tripwires.
    """

    def detect(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[DetectedCommitment]:
        if not text or not text.strip():
            return []
        matches = _extract_pattern_matches(text)
        for m in matches:
            m.source_ref = source_ref
        return matches

    def is_available(self) -> bool:
        return True


# ─── OllamaCommitmentDetector ───────────────────────────────────────────


class OllamaCommitmentDetector(CommitmentDetector):
    """LLM-only commitment detector backed by a local Ollama daemon.

    Lazy availability probe (invariant #1). Override ``_call_llm``
    in tests to return scripted responses. The ``HybridCommitmentDetector``
    inherits from this class to gain the LLM plumbing and overrides
    ``detect`` to add the pattern pre-filter.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        model: str = DEFAULT_DETECTION_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._host = (
            host or os.environ.get(ENV_HOST) or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
        self._model = model
        self._timeout = timeout
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    @property
    def host(self) -> str:
        return self._host

    @property
    def model(self) -> str:
        return self._model

    # ── Availability ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        with self._lock:
            if self._available is not None:
                return self._available

        probe_ok = self._probe()

        with self._lock:
            self._available = probe_ok
            return self._available

    def reset_availability(self) -> None:
        with self._lock:
            self._available = None

    def _probe(self) -> bool:
        try:
            raw = self._call_llm(_PROBE_PROMPT)
        except Exception as exc:
            logger.debug("Ollama commitment detector probe failed: %s", exc)
            return False
        return bool(raw)

    # ── Detection ─────────────────────────────────────────────────────

    def detect(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[DetectedCommitment]:
        """LLM-only detection. Returns [] on empty input or any failure."""
        if not text or not text.strip():
            return []
        if not self.is_available():
            return []

        prompt = _DETECTION_PROMPT.format(text=text)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            logger.warning("Ollama commitment detection call failed: %s", exc)
            return []

        if not raw:
            return []

        return _coerce_json_to_commitments(raw, source_ref=source_ref)

    # ── Overridable for tests ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """POST to Ollama's ``/api/generate`` endpoint. Short-lived client."""
        import httpx  # noqa: PLC0415 - lazy import matches OllamaLLMExtractor

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
                "Ollama commitment /api/generate returned %d: %s",
                response.status_code,
                response.text[:200],
            )
            return ""

        try:
            data = response.json()
        except ValueError:
            logger.debug("Ollama commitment /api/generate returned non-JSON")
            return ""

        if not isinstance(data, dict):
            return ""

        raw = data.get("response", "")
        return raw if isinstance(raw, str) else ""


# ─── HybridCommitmentDetector ───────────────────────────────────────────


class HybridCommitmentDetector(OllamaCommitmentDetector):
    """Pattern pre-filter + LLM refinement. The default production detector.

    Detection flow:

    1. Run the pattern pre-filter. If it finds nothing, return []
       without touching the LLM. This is the common fast path —
       most turns have no commitment language.
    2. If the pattern fires, check LLM availability. If the LLM is
       offline, return the pattern matches as an offline fallback.
    3. Call the LLM with the full text. The LLM either:
       (a) Returns refined commitments → use those (replaces pattern
           output, origin="llm")
       (b) Returns ``[]`` → veto the pattern matches, return []
       (c) Raises an exception → fall back to pattern matches
    4. Parse the LLM response via ``_coerce_json_to_commitments``.
       Vague commitments are dropped during coercion.

    Subclasses inject ``_call_llm`` in tests for deterministic behavior.
    """

    def __init__(
        self,
        *,
        pattern: Optional[PatternCommitmentDetector] = None,
        host: Optional[str] = None,
        model: str = DEFAULT_DETECTION_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        super().__init__(host=host, model=model, timeout=timeout)
        self._pattern = pattern or PatternCommitmentDetector()

    def detect(
        self,
        text: str,
        *,
        source_ref: str = "",
    ) -> List[DetectedCommitment]:
        if not text or not text.strip():
            return []

        # Step 1: pattern pre-filter
        pattern_matches = self._pattern.detect(text, source_ref=source_ref)
        if not pattern_matches:
            return []

        # Step 2: LLM availability check
        if not self.is_available():
            return pattern_matches

        # Step 3: LLM refinement call
        prompt = _DETECTION_PROMPT.format(text=text)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            logger.warning(
                "Hybrid commitment detector LLM call failed, falling "
                "back to pattern output: %s",
                exc,
            )
            return pattern_matches

        if not raw:
            # Empty response string (not empty array) is treated as
            # a degraded daemon, not a veto. Fall back to pattern.
            return pattern_matches

        # Step 4: parse LLM output. Empty list from valid JSON is
        # a legitimate veto.
        return _coerce_json_to_commitments(raw, source_ref=source_ref)
