"""Reusable spend-control primitive for agent payment tools.

One ``BudgetPolicy`` instance = one session. State is in-memory only —
creating a new policy resets the session.

Callers hold ``policy.lock()`` across check → approve → pay → commit so
concurrent payment attempts on a shared policy can't both pass a cap check
when only one would fit.
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Callable


class BudgetPolicy:
    """Per-call + per-session spend caps with optional human-in-the-loop gate.

    One policy = one session. Caps are read on every ``check()``, so you can
    tune them at runtime, but do it from the thread that owns the policy.

    The approval callback is invoked while the policy lock is held, so it
    must not block on I/O that waits for another thread touching this same
    policy. If you need an interactive approval UI, run it on a separate
    thread/process and return a bool synchronously here.

    >>> from decimal import Decimal
    >>> p = BudgetPolicy(max_per_call=Decimal("0.10"))
    >>> p.check(Decimal("0.05")) is None
    True
    >>> p.check(Decimal("0.20"))["reason"]
    'max_per_call'
    >>> p.check(Decimal("-1"))["reason"]
    'invalid_price'
    """

    def __init__(
        self,
        *,
        max_per_call: Decimal = Decimal("0.10"),
        max_per_session: Decimal = Decimal("5.00"),
        human_approval_above: Decimal | None = None,
        approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.max_per_call = max_per_call
        self.max_per_session = max_per_session
        self.human_approval_above = human_approval_above
        self.approval_callback = approval_callback
        self._session_spent: Decimal = Decimal("0")
        self._lock: threading.RLock = threading.RLock()

    @property
    def session_spent(self) -> Decimal:
        with self._lock:
            return self._session_spent

    def lock(self) -> threading.RLock:
        return self._lock

    def check(self, price: Decimal) -> dict[str, Any] | None:
        """Return None if the payment fits the budget, else an error dict.

        Caller must hold ``self.lock()`` so the read of ``_session_spent`` and
        the subsequent ``commit()`` form an atomic sequence. Rejects negative,
        zero, and non-finite prices so a misbehaving gate can't bypass caps
        with ``"-1"`` or ``"NaN"``.

        >>> from decimal import Decimal
        >>> p = BudgetPolicy(
        ...     max_per_call=Decimal("1.00"), max_per_session=Decimal("1.00"),
        ... )
        >>> p.check(Decimal("0.50")) is None
        True
        >>> p.commit(Decimal("0.50"))
        >>> p.check(Decimal("0.60"))["reason"]
        'max_per_session'
        """
        if not price.is_finite() or price <= 0:
            return {
                "error": "budget_exceeded",
                "reason": "invalid_price",
                "price_usd": str(price),
            }
        if price > self.max_per_call:
            return {
                "error": "budget_exceeded",
                "reason": "max_per_call",
                "price_usd": str(price),
                "limit_usd": str(self.max_per_call),
            }
        with self._lock:
            if self._session_spent + price > self.max_per_session:
                return {
                    "error": "budget_exceeded",
                    "reason": "max_per_session",
                    "session_spent_usd": str(self._session_spent),
                    "price_usd": str(price),
                    "limit_usd": str(self.max_per_session),
                }
        return None

    def requires_approval(self, price: Decimal) -> bool:
        """True when ``price`` exceeds the configured human-approval threshold.

        >>> from decimal import Decimal
        >>> p = BudgetPolicy(human_approval_above=Decimal("0.10"))
        >>> p.requires_approval(Decimal("0.10"))
        False
        >>> p.requires_approval(Decimal("0.11"))
        True
        """
        return (
            self.human_approval_above is not None
            and price > self.human_approval_above
        )

    def request_approval(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        """Return None when approved (or not needed), or an error/declined dict.

        ``plan`` must include ``price_usd`` as a string. Other fields are
        passed through to the callback unchanged.

        >>> from decimal import Decimal
        >>> approved = BudgetPolicy(
        ...     human_approval_above=Decimal("0.01"),
        ...     approval_callback=lambda plan: True,
        ... )
        >>> approved.request_approval({"price_usd": "0.05"}) is None
        True
        >>> declined = BudgetPolicy(
        ...     human_approval_above=Decimal("0.01"),
        ...     approval_callback=lambda plan: False,
        ... )
        >>> declined.request_approval({"price_usd": "0.05"})["status"]
        'declined'
        """
        price = Decimal(plan["price_usd"])
        if not self.requires_approval(price):
            return None
        if self.approval_callback is None:
            return {
                "error": "approval_required",
                "reason": "no_callback_configured",
                "price_usd": str(price),
                "threshold_usd": str(self.human_approval_above),
            }
        if not bool(self.approval_callback(plan)):
            return {
                "status": "declined",
                "reason": "approval_callback_rejected",
                "price_usd": str(price),
            }
        return None

    def commit(self, price: Decimal) -> None:
        """Add ``price`` to session-spend total.

        Takes the RLock internally so a miswired caller can't tear the
        read-modify-write. Safe to call inside an outer ``with self.lock():``
        block because the lock is reentrant.

        >>> from decimal import Decimal
        >>> p = BudgetPolicy()
        >>> p.commit(Decimal("0.03"))
        >>> p.commit(Decimal("0.07"))
        >>> p.session_spent
        Decimal('0.10')
        """
        with self._lock:
            self._session_spent += price
