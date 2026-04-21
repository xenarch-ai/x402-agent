"""Unit tests for the neutral ``BudgetPolicy`` (XEN-167 / PR 6b).

These live alongside the framework-agnostic ``x402_agent`` subpackage so
they travel with it when the package is promoted to its own repo. No
Xenarch imports; no framework deps.
"""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from x402_agent import BudgetPolicy


class TestDefaults:
    def test_defaults_block_big_single_call(self) -> None:
        p = BudgetPolicy()
        err = p.check(Decimal("1.00"))
        assert err is not None
        assert err["reason"] == "max_per_call"

    def test_session_spent_starts_at_zero(self) -> None:
        assert BudgetPolicy().session_spent == Decimal("0")


class TestInvalidPrices:
    """Caps assume sanitised input. A misbehaving gate must not pass ``-1``
    or ``NaN`` through; ``check`` rejects them before the cap compare so a
    negative price can never decrement the session budget."""

    @pytest.mark.parametrize(
        "raw", ["0", "-0.01", "-1", "NaN", "Infinity", "-Infinity"]
    )
    def test_non_positive_or_non_finite_rejected(self, raw: str) -> None:
        p = BudgetPolicy()
        err = p.check(Decimal(raw))
        assert err is not None
        assert err["reason"] == "invalid_price"


class TestSessionAccumulation:
    def test_commit_advances_session_spend(self) -> None:
        p = BudgetPolicy(
            max_per_call=Decimal("1.00"), max_per_session=Decimal("2.00")
        )
        p.commit(Decimal("0.30"))
        p.commit(Decimal("0.50"))
        assert p.session_spent == Decimal("0.80")

    def test_session_cap_is_enforced_after_prior_commits(self) -> None:
        p = BudgetPolicy(
            max_per_call=Decimal("1.00"), max_per_session=Decimal("1.00")
        )
        p.commit(Decimal("0.90"))
        err = p.check(Decimal("0.20"))
        assert err is not None
        assert err["reason"] == "max_per_session"


class TestApproval:
    def test_above_threshold_with_no_callback_errors(self) -> None:
        p = BudgetPolicy(human_approval_above=Decimal("0.01"))
        out = p.request_approval({"price_usd": "0.05"})
        assert out is not None
        assert out["error"] == "approval_required"

    def test_callback_decline_becomes_declined_status(self) -> None:
        p = BudgetPolicy(
            human_approval_above=Decimal("0.01"),
            approval_callback=lambda plan: False,
        )
        out = p.request_approval({"price_usd": "0.05"})
        assert out is not None
        assert out["status"] == "declined"

    def test_under_threshold_no_callback_needed(self) -> None:
        called: list[dict[str, str]] = []

        p = BudgetPolicy(
            human_approval_above=Decimal("1.00"),
            approval_callback=lambda plan: called.append(plan) or True,
        )
        assert p.request_approval({"price_usd": "0.50"}) is None
        assert called == []


class TestLockReentrancy:
    """``commit`` must be safe to call while the caller already holds
    ``policy.lock()``. If the underlying lock were non-reentrant, the pay
    loop would deadlock on its own commit."""

    def test_commit_inside_lock_does_not_deadlock(self) -> None:
        p = BudgetPolicy()

        done = threading.Event()

        def worker() -> None:
            with p.lock():
                p.commit(Decimal("0.01"))
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=1.0)
        assert done.is_set(), "commit-inside-lock deadlocked"
        assert p.session_spent == Decimal("0.01")
