"""Unit tests for plugins/memory/claudia/budget.py (Phase 2B.5).

Covered:

- ``BudgetState`` dataclass defaults
- ``BudgetDecision`` dataclass defaults and fields
- ``decide_budget`` thresholds:
    * None remaining → default decision (full budget)
    * below CRITICAL → skip all cognitive work, minimal prefetch
    * between CRITICAL and LOW → reduced prefetch, no skipping
    * above LOW → default decision
- ``update_budget_state`` from kwargs
- Constants sanity (CRITICAL < LOW, both > 0)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plugins.memory.claudia.budget import (
    CHARS_PER_TOKEN,
    CRITICAL_TOKEN_THRESHOLD,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_PREFETCH_TOKEN_BUDGET,
    LOW_TOKEN_THRESHOLD,
    BudgetDecision,
    BudgetState,
    decide_budget,
    estimate_tokens,
    truncate_to_budget,
    update_budget_state,
)


NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ─── Dataclass shapes ───────────────────────────────────────────────────


class TestBudgetState:
    def test_defaults(self):
        s = BudgetState()
        assert s.remaining_tokens is None
        assert s.turn_number == 0
        assert s.last_updated is None


class TestBudgetDecision:
    def test_defaults(self):
        d = BudgetDecision()
        assert d.prefetch_limit == DEFAULT_PREFETCH_LIMIT
        assert d.prefetch_budget_tokens == DEFAULT_PREFETCH_TOKEN_BUDGET
        assert d.skip_extraction is False
        assert d.skip_detection is False


# ─── Threshold constants sanity ────────────────────────────────────────


class TestThresholdConstants:
    def test_critical_less_than_low(self):
        assert CRITICAL_TOKEN_THRESHOLD < LOW_TOKEN_THRESHOLD

    def test_both_positive(self):
        assert CRITICAL_TOKEN_THRESHOLD > 0
        assert LOW_TOKEN_THRESHOLD > 0

    def test_default_prefetch_limit_positive(self):
        assert DEFAULT_PREFETCH_LIMIT >= 1

    def test_default_prefetch_budget_positive(self):
        assert DEFAULT_PREFETCH_TOKEN_BUDGET >= 0


# ─── decide_budget ──────────────────────────────────────────────────────


class TestDecideBudget:
    def test_none_remaining_returns_default(self):
        """No budget info → full budget, nothing skipped."""
        state = BudgetState(remaining_tokens=None)
        decision = decide_budget(state)
        assert decision.prefetch_limit == DEFAULT_PREFETCH_LIMIT
        assert decision.prefetch_budget_tokens == DEFAULT_PREFETCH_TOKEN_BUDGET
        assert decision.skip_extraction is False
        assert decision.skip_detection is False

    def test_above_low_returns_default(self):
        state = BudgetState(remaining_tokens=LOW_TOKEN_THRESHOLD + 1000)
        decision = decide_budget(state)
        assert decision.prefetch_limit == DEFAULT_PREFETCH_LIMIT
        assert decision.skip_extraction is False

    def test_at_critical_skips_cognitive(self):
        state = BudgetState(remaining_tokens=CRITICAL_TOKEN_THRESHOLD - 1)
        decision = decide_budget(state)
        assert decision.skip_extraction is True
        assert decision.skip_detection is True
        assert decision.prefetch_limit < DEFAULT_PREFETCH_LIMIT

    def test_at_low_reduces_prefetch(self):
        """Between CRITICAL and LOW: reduce prefetch but keep cognitive."""
        state = BudgetState(
            remaining_tokens=CRITICAL_TOKEN_THRESHOLD + 1
        )
        decision = decide_budget(state)
        assert decision.skip_extraction is False
        assert decision.skip_detection is False
        assert decision.prefetch_limit <= DEFAULT_PREFETCH_LIMIT

    def test_zero_remaining_treated_as_critical(self):
        state = BudgetState(remaining_tokens=0)
        decision = decide_budget(state)
        assert decision.skip_extraction is True

    def test_negative_remaining_treated_as_critical(self):
        """Defensive: negative budget (shouldn't happen) → critical."""
        state = BudgetState(remaining_tokens=-100)
        decision = decide_budget(state)
        assert decision.skip_extraction is True


# ─── update_budget_state ────────────────────────────────────────────────


# ─── Token estimation (Phase 2B.6) ──────────────────────────────────────


class TestEstimateTokens:
    def test_empty_returns_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_short_text(self):
        """A 12-char string is ~3 tokens (12/4)."""
        assert estimate_tokens("hello world!") == 3

    def test_rounds_up(self):
        """5 chars is 2 tokens (ceil(5/4))."""
        assert estimate_tokens("hello") == 2

    def test_monotonic(self):
        """Longer text always has >= token count."""
        a = estimate_tokens("hello")
        b = estimate_tokens("hello world")
        assert b >= a


# ─── truncate_to_budget ─────────────────────────────────────────────────


class TestTruncateToBudget:
    def test_under_budget_returns_unchanged(self):
        text = "\n".join(["line 1", "line 2", "line 3"])
        assert truncate_to_budget(text, max_tokens=100) == text

    def test_zero_budget_returns_empty(self):
        assert truncate_to_budget("hello world", max_tokens=0) == ""
        assert truncate_to_budget("hello world", max_tokens=-1) == ""

    def test_empty_input(self):
        assert truncate_to_budget("", max_tokens=100) == ""

    def test_drops_lines_from_bottom(self):
        """Over-budget text drops lines from the bottom, not the top.

        Uses a realistic budget (20 tokens) so the header + some
        bullets survive. Smaller budgets can't fit the truncation
        marker plus any content, and return empty.
        """
        text = "## Claudia Memory\n- first fact\n- second fact\n- third fact\n- fourth fact\n- fifth fact"
        # Total ~90 chars → ~23 tokens. Budget 20 tokens → drop 1-2 bullets.
        result = truncate_to_budget(text, max_tokens=20)
        assert "## Claudia Memory" in result
        assert "first fact" in result
        assert "truncated" in result.lower()
        # Bottom bullets should be gone
        assert "fifth fact" not in result

    def test_marker_appended_on_truncation(self):
        text = "\n".join([f"- item {i}" for i in range(20)])
        result = truncate_to_budget(text, max_tokens=30)
        assert "truncated" in result.lower()

    def test_tiny_budget_returns_empty(self):
        """A budget smaller than one line + marker returns empty."""
        text = "some longer text that cannot fit"
        result = truncate_to_budget(text, max_tokens=1)
        assert result == ""

    def test_preserves_header_line(self):
        """Whole-line truncation keeps the header when budget permits."""
        text = "## Claudia Memory\n- a\n- b\n- c\n- d\n- e\n- f\n- g"
        # ~38 chars → ~10 tokens. Budget 12 should keep header + marker
        # and some bullets.
        result = truncate_to_budget(text, max_tokens=12)
        assert "## Claudia Memory" in result


class TestUpdateBudgetState:
    def test_none_kwargs_returns_unchanged(self):
        state = BudgetState(
            remaining_tokens=5000, turn_number=3, last_updated=NOW
        )
        new = update_budget_state(state, {})
        assert new.remaining_tokens == 5000
        assert new.turn_number == 3

    def test_remaining_tokens_updated(self):
        state = BudgetState()
        new = update_budget_state(
            state, {"remaining_tokens": 4200}, now=NOW, turn_number=5
        )
        assert new.remaining_tokens == 4200
        assert new.turn_number == 5
        assert new.last_updated == NOW

    def test_invalid_remaining_tokens_ignored(self):
        state = BudgetState()
        new = update_budget_state(
            state, {"remaining_tokens": "not a number"}, now=NOW
        )
        assert new.remaining_tokens is None

    def test_partial_update_preserves_other_fields(self):
        state = BudgetState(remaining_tokens=1000, turn_number=2)
        new = update_budget_state(
            state, {"remaining_tokens": 500}, now=NOW, turn_number=3
        )
        assert new.remaining_tokens == 500
        assert new.turn_number == 3
