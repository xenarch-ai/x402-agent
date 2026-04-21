"""Unit tests for the framework-agnostic helpers in ``x402_agent._helpers``.

Every function under test is pure (``is_public_host`` hits DNS but is
covered against fixed literals). Tests live outside ``xenarch`` so the
package is self-testable in its upstream repo.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from x402.schemas import PaymentRequired, PaymentRequirements

from x402_agent import (
    BudgetPolicy,
    DEFAULT_NETWORK,
    budget_hint_exceeds,
    is_public_host,
    price_usd,
    select_accept,
)
from x402_agent._helpers import split_host_path, truncate_body


USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


def _req(
    *,
    scheme: str = "exact",
    network: str = DEFAULT_NETWORK,
    amount: str = "10000",
    extra: dict[str, Any] | None = None,
) -> PaymentRequirements:
    return PaymentRequirements(
        scheme=scheme,
        network=network,
        asset=USDC,
        amount=amount,
        payTo="0x0000000000000000000000000000000000000001",
        maxTimeoutSeconds=60,
        extra=extra or {"name": "USD Coin", "version": "2"},
    )


class TestPriceUsd:
    def test_default_usdc_decimals(self) -> None:
        # 10000 in 6-decimal USDC is $0.01.
        assert price_usd(_req(amount="10000")) == Decimal("0.01")

    def test_extra_decimals_override(self) -> None:
        # Publisher advertises an 18-decimal asset; helper honours extra.
        # 10^17 atomic units at 18 decimals = $0.1.
        r = _req(amount="1" + "0" * 17, extra={"decimals": 18})
        assert price_usd(r) == Decimal("0.1")

    def test_non_int_decimals_falls_back_to_default(self) -> None:
        # Garbage in `extra.decimals` must not crash — fall back to 6.
        r = _req(amount="10000", extra={"decimals": "six"})
        assert price_usd(r) == Decimal("0.01")


class TestSelectAccept:
    def test_prefers_exact_match_on_base(self) -> None:
        pr = PaymentRequired(
            x402Version=2,
            error="payment_required",
            accepts=[
                _req(network="eip155:1"),  # Ethereum
                _req(network=DEFAULT_NETWORK),  # Base — prefer this
            ],
        )
        chosen = select_accept(pr)
        assert chosen is not None
        assert chosen.network == DEFAULT_NETWORK

    def test_falls_back_to_any_eip155(self) -> None:
        pr = PaymentRequired(
            x402Version=2,
            error="payment_required",
            accepts=[_req(network="eip155:1")],
        )
        chosen = select_accept(pr)
        assert chosen is not None
        assert chosen.network == "eip155:1"

    def test_rejects_legacy_non_caip2_network(self) -> None:
        # V1 legacy string "base" isn't CAIP-2 and isn't registered on
        # the SDK's EVM client. Must return None so callers surface
        # ``no_supported_scheme`` rather than silently signing with the
        # wrong chain ID.
        pr = PaymentRequired(
            x402Version=2,
            error="payment_required",
            accepts=[_req(network="base")],
        )
        assert select_accept(pr) is None


class TestTruncateBody:
    def test_under_limit_untouched(self) -> None:
        assert truncate_body("abc", 100) == "abc"

    def test_over_limit_appends_ellipsis(self) -> None:
        assert truncate_body("abcdef", 3) == "abc…"


class TestSplitHostPath:
    def test_strips_userinfo(self) -> None:
        # ``user:pass@`` must never leak into logs or pay.json lookups.
        host, path = split_host_path("https://u:p@example.com/a")
        assert host == "example.com"
        assert path == "/a"

    def test_keeps_explicit_port(self) -> None:
        host, _ = split_host_path("http://example.com:8080/x")
        assert host == "example.com:8080"

    def test_empty_path_becomes_root(self) -> None:
        _, path = split_host_path("https://example.com")
        assert path == "/"


class TestIsPublicHost:
    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "127.0.0.1",
            "10.0.0.1",  # RFC1918
            "192.168.1.1",  # RFC1918
            "169.254.169.254",  # AWS/GCP IMDS — the canonical SSRF target
            "::1",  # IPv6 loopback
            "",  # empty
        ],
    )
    def test_private_and_metadata_ranges_blocked(self, host: str) -> None:
        assert is_public_host(host) is False

    def test_unresolvable_host_blocked(self) -> None:
        # A syntactically-valid name that does not resolve must return
        # False (fail closed), not raise — the pay loop would otherwise
        # crash instead of returning ``unsafe_host``.
        assert (
            is_public_host("this-host-definitely-does-not-exist.invalid")
            is False
        )


class TestBudgetHintExceeds:
    def test_per_call_hint_over_cap_refuses(self) -> None:
        out = budget_hint_exceeds(
            {"recommended_max_per_call": "2.00"},
            BudgetPolicy(max_per_call=Decimal("0.10")),
        )
        assert out is not None
        assert out["reason"] == "recommended_max_per_call"

    def test_non_string_hint_falls_through(self) -> None:
        # Spec says string. A raw float is "no guidance".
        out = budget_hint_exceeds(
            {"recommended_max_per_call": 2.0},
            BudgetPolicy(max_per_call=Decimal("0.10")),
        )
        assert out is None

    def test_nan_hint_falls_through(self) -> None:
        out = budget_hint_exceeds(
            {"recommended_max_per_call": "NaN"},
            BudgetPolicy(max_per_call=Decimal("0.10")),
        )
        assert out is None

    def test_policy_is_duck_typed(self) -> None:
        # Helper must work against any budget-like object so adapter
        # authors can use their framework's own policy class.
        class _DuckPolicy:
            max_per_call = Decimal("0.10")
            max_per_session = Decimal("1.00")

        out = budget_hint_exceeds(
            {"recommended_max_per_session": "5.00"}, _DuckPolicy()
        )
        assert out is not None
        assert out["reason"] == "recommended_max_per_session"
