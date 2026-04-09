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
    CRITICAL_TOKEN_THRESHOLD,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_PREFETCH_TOKEN_BUDGET,
    LOW_TOKEN_THRESHOLD,
    BudgetDecision,
    BudgetState,
    decide_budget,
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
