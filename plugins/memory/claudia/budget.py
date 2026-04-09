"""Cost governance for Claudia memory (Phase 2B.5).

The ABC's ``on_turn_start(turn_number, message, **kwargs)`` hook
passes ``remaining_tokens`` to providers. Per the ABC contract,
providers can READ this signal but cannot BLOCK the call — real
enforcement happens in ``run_agent.py``. Claudia uses the signal
to gracefully degrade her own resource usage:

- When the budget is **critical** (below CRITICAL_TOKEN_THRESHOLD),
  skip cognitive work entirely (entity extraction and commitment
  detection) and return the minimum viable prefetch payload (3
  memories, ~500 tokens). The current turn still gets context,
  just compact context.
- When the budget is **low** (between CRITICAL and LOW), reduce
  the prefetch limit but continue cognitive work. Most sessions
  are fine here; this is the "tighten the belt" mode.
- Otherwise, the default decision applies (10 memories, ~1500
  tokens, full cognitive work).

Design principles:

- **Stateless decision function.** ``decide_budget(state)`` is a
  pure function of the state object. No side effects, no clock
  reads. Tests drive it with synthetic states.
- **Permissive on missing info.** If ``remaining_tokens`` is None
  (the agent didn't pass it, or we haven't seen a turn yet), the
  default decision applies. Missing info defaults to "assume
  full budget".
- **Read-only on the provider.** The provider mutates
  ``_budget_state`` ONLY in ``on_turn_start``. Every other hook
  reads the current state and calls ``decide_budget`` to get a
  fresh decision. This keeps the hot path (prefetch, sync_turn)
  lock-free aside from the dict lookup.

Public API:

- ``BudgetState`` dataclass (remaining_tokens, turn_number, last_updated)
- ``BudgetDecision`` dataclass (prefetch_limit, prefetch_budget_tokens,
  skip_extraction, skip_detection)
- ``decide_budget(state)`` → BudgetDecision
- ``update_budget_state(state, kwargs, *, now, turn_number)`` → BudgetState

Reference: agent/memory_provider.py on_turn_start ABC contract,
plans/phase-2b-handoff.md Phase 2B.5 scope notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ─── Constants ──────────────────────────────────────────────────────────


#: Remaining-token budget below which the provider skips ALL
#: cognitive work and returns minimal prefetch. "I'm almost out
#: of room, cut everything that isn't essential."
CRITICAL_TOKEN_THRESHOLD = 2_000

#: Remaining-token budget below which the provider reduces prefetch
#: size but keeps cognitive work running. "I have room but not a
#: lot; be more selective."
LOW_TOKEN_THRESHOLD = 5_000

#: Default number of memories the prefetch returns under normal
#: budget. Matches the ``limit=10`` previously hardcoded in
#: ``provider.prefetch``.
DEFAULT_PREFETCH_LIMIT = 10

#: Target token budget for prefetch output under normal conditions.
#: This is the provider's contribution slice of the ~9300-token
#: prompt ceiling discussed in the Phase 2A.1 design doc.
DEFAULT_PREFETCH_TOKEN_BUDGET = 1_500

#: Reduced prefetch limit when budget is low.
LOW_PREFETCH_LIMIT = 5
LOW_PREFETCH_TOKEN_BUDGET = 1_000

#: Minimal prefetch parameters at critical budget.
CRITICAL_PREFETCH_LIMIT = 3
CRITICAL_PREFETCH_TOKEN_BUDGET = 500


# ─── Dataclasses ────────────────────────────────────────────────────────


@dataclass
class BudgetState:
    """Snapshot of the current turn's token budget.

    Populated from ``on_turn_start(**kwargs)`` and read by every
    subsequent hook during the turn. Reset on each turn.
    """

    remaining_tokens: Optional[int] = None
    turn_number: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class BudgetDecision:
    """The decision derived from a BudgetState.

    Consumers use this to parameterize their work without
    re-implementing the threshold logic.
    """

    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    prefetch_budget_tokens: int = DEFAULT_PREFETCH_TOKEN_BUDGET
    skip_extraction: bool = False
    skip_detection: bool = False


# ─── decide_budget ──────────────────────────────────────────────────────


def decide_budget(state: BudgetState) -> BudgetDecision:
    """Map a BudgetState to a BudgetDecision.

    Pure function: same state always yields the same decision.

    - None remaining → default (assume full budget)
    - remaining < CRITICAL → skip cognitive, minimal prefetch
    - remaining < LOW → reduce prefetch, keep cognitive
    - otherwise → default
    """
    remaining = state.remaining_tokens

    if remaining is None:
        return BudgetDecision()

    if remaining < CRITICAL_TOKEN_THRESHOLD:
        return BudgetDecision(
            prefetch_limit=CRITICAL_PREFETCH_LIMIT,
            prefetch_budget_tokens=CRITICAL_PREFETCH_TOKEN_BUDGET,
            skip_extraction=True,
            skip_detection=True,
        )

    if remaining < LOW_TOKEN_THRESHOLD:
        return BudgetDecision(
            prefetch_limit=LOW_PREFETCH_LIMIT,
            prefetch_budget_tokens=LOW_PREFETCH_TOKEN_BUDGET,
            skip_extraction=False,
            skip_detection=False,
        )

    return BudgetDecision()


# ─── update_budget_state ────────────────────────────────────────────────


def update_budget_state(
    state: BudgetState,
    kwargs: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    turn_number: Optional[int] = None,
) -> BudgetState:
    """Build a new BudgetState from ``on_turn_start`` kwargs.

    ``remaining_tokens`` is coerced to int if present and numeric;
    otherwise left as None. ``turn_number`` and ``now`` are passed
    through from the caller. Returns a fresh BudgetState — the
    old one is not mutated.
    """
    remaining = state.remaining_tokens
    if "remaining_tokens" in kwargs:
        raw = kwargs["remaining_tokens"]
        if isinstance(raw, (int, float)):
            remaining = int(raw)
        else:
            remaining = None  # invalid value → reset to unknown

    return BudgetState(
        remaining_tokens=remaining,
        turn_number=turn_number if turn_number is not None else state.turn_number,
        last_updated=now or datetime.now(timezone.utc),
    )


# ─── Token estimation and truncation (Phase 2B.6) ───────────────────────


#: Approximate characters per token. Tokenizers vary (BPE, SentencePiece,
#: word-piece) but this is the rough middle-of-the-road for English.
#: Callers that need exact counts should tokenize through the actual
#: model, but for budget decisions this is close enough — we'd rather
#: over-estimate and truncate a bit more than over-run the budget.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without running a tokenizer.

    Uses ``len(text) / CHARS_PER_TOKEN`` as an approximation, rounded
    up. Returns 0 for empty or None input. Deliberately approximate:
    exact counts would require the model's real tokenizer, which is
    overkill for budget decisions that have ±10% tolerance anyway.
    """
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def truncate_to_budget(
    text: str,
    max_tokens: int,
    *,
    truncated_marker: str = "\n- ... (truncated)",
) -> str:
    """Truncate ``text`` so the estimated token count fits within
    ``max_tokens``.

    Truncation is line-oriented — we drop whole lines from the END
    of the text until the result (plus the marker) fits under budget.
    The marker is appended only when truncation actually happened.

    Line-oriented truncation is a good fit for the prefetch output
    format, which is one bullet per memory plus a header line.
    Dropping whole bullets preserves the structure the LLM expects.

    Returns the text unchanged if already under budget. Returns an
    empty string if ``max_tokens`` is 0/negative or if no line
    (plus marker) can fit within the budget.
    """
    if max_tokens <= 0:
        return ""
    if not text:
        return text

    if estimate_tokens(text) <= max_tokens:
        return text

    lines = text.split("\n")

    # Drop lines from the bottom until ``content + marker`` fits.
    # The marker is included in the measurement so we don't overrun
    # after appending it. If even a single line + marker exceeds
    # the budget, we'll drain the list completely and return empty.
    while lines:
        candidate = "\n".join(lines) + truncated_marker
        if estimate_tokens(candidate) <= max_tokens:
            return candidate
        lines.pop()

    # No line combination fits. Return empty rather than emitting
    # a marker-only stub (cleaner for the caller).
    return ""
