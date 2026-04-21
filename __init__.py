"""x402-agent — framework-agnostic payer on top of the x402 protocol SDK.

Layer 1 of the three-layer agent-payments architecture:

    Layer 3 — Xenarch commercial flavors (receipts, reputation, etc.)
    Layer 2 — Per-framework thin adapters (LangChain, CrewAI, AutoGen, LangGraph)
    Layer 1 — THIS PACKAGE. Neutral `x402Payer` + `BudgetPolicy` + SSRF guard
    Layer 0 — Coinbase ``x402`` SDK (protocol primitives)

This package has zero framework dependencies. Framework adapters wrap it;
they never leak framework types into this namespace.

Subpackaged inside ``xenarch-sdks`` for convenience during Phase 0 of the
rollout plan (see ``Information/design/x402-agent-upstream-plan.md``).
Phase 1 promotes this subpackage to its own public repo + PyPI release.
"""

from x402_agent._budget import BudgetPolicy
from x402_agent._helpers import (
    DEFAULT_NETWORK,
    DEFAULT_SCHEME,
    X_PAYMENT_HEADER,
    X_PAYMENT_RESPONSE_HEADER,
    budget_hint_exceeds,
    is_public_host,
    is_public_host_async,
    price_usd,
    select_accept,
)
from x402_agent._payer import X402Payer

# v0.1.0 stable surface. Adapter authors should import only from here.
# Everything in ``x402_agent._helpers`` and ``x402_agent._payer`` is
# accessible but not a stability promise — we may rename it.
__all__ = [
    "BudgetPolicy",
    "DEFAULT_NETWORK",
    "DEFAULT_SCHEME",
    "X402Payer",
    "X_PAYMENT_HEADER",
    "X_PAYMENT_RESPONSE_HEADER",
    "budget_hint_exceeds",
    "is_public_host",
    "is_public_host_async",
    "price_usd",
    "select_accept",
]
