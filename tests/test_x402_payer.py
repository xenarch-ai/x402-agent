"""Integration-ish tests for the neutral ``X402Payer`` (XEN-167 / PR 6b).

These exercise the full sync + async pay loop through an ``httpx``
mock transport — no Xenarch subclass, no receipts, no reputation. The
goal is to prove the neutral core is self-sufficient for a framework
adapter (LangChain, CrewAI, AutoGen, LangGraph) to depend on.
"""

from __future__ import annotations

import base64
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest
from eth_account import Account

from x402_agent import BudgetPolicy, X402Payer


USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


def _make_402_body(amount: str = "10000") -> dict[str, Any]:
    return {
        "x402Version": 2,
        "error": "payment_required",
        "accepts": [
            {
                "scheme": "exact",
                "network": "eip155:8453",
                "asset": USDC,
                "amount": amount,
                "payTo": "0x0000000000000000000000000000000000000001",
                "maxTimeoutSeconds": 60,
                "extra": {"name": "USD Coin", "version": "2"},
            }
        ],
    }


def _settle_header(tx: str = "0xfeedface") -> str:
    return base64.b64encode(
        json.dumps(
            {"success": True, "transaction": tx, "network": "eip155:8453"}
        ).encode()
    ).decode()


def _payer(**overrides: Any) -> X402Payer:
    defaults: dict[str, Any] = {
        "private_key": Account.create().key.hex(),
        "budget_policy": BudgetPolicy(
            max_per_call=Decimal("0.05"),
            max_per_session=Decimal("1.00"),
        ),
        "discover_via_pay_json": False,
    }
    defaults.update(overrides)
    return X402Payer(**defaults)


def _install_sync_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Any
) -> None:
    transport = httpx.MockTransport(handler)
    real = httpx.Client

    class _MC(real):  # type: ignore[misc,valid-type]
        def __init__(
            self, transport: httpx.MockTransport, **kw: Any
        ) -> None:
            super().__init__(transport=transport, **kw)

    def _factory(*a: Any, **kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return _MC(transport=transport, **kw)

    monkeypatch.setattr("x402_agent._payer.httpx.Client", _factory)


def _install_async_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Any
) -> None:
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    class _MAC(real):  # type: ignore[misc,valid-type]
        def __init__(
            self, transport: httpx.MockTransport, **kw: Any
        ) -> None:
            super().__init__(transport=transport, **kw)

    def _factory(*a: Any, **kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return _MAC(transport=transport, **kw)

    monkeypatch.setattr("x402_agent._payer.httpx.AsyncClient", _factory)


class TestSyncHappyPath:
    def test_402_challenge_then_paid_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if "x-payment" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="paid body",
                headers={"X-PAYMENT-RESPONSE": _settle_header()},
            )

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/article/1")
        assert result["success"] is True
        assert result["body"] == "paid body"
        assert result["amount_usd"] == "0.01"
        # Session budget must advance by exactly the paid amount.
        assert result["session_spent_usd"] == "0.01"

    def test_non_402_response_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Free resource (200 up front). Payer reports no_payment_required
        # rather than trying to force-pay.
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="free content")

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/free")
        assert result["status"] == "no_payment_required"
        assert result["body"] == "free content"


class TestSyncErrorShapes:
    def test_unparseable_402_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(402, text="not json")

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/broken")
        assert result["error"] == "x402_parse_failed"

    def test_no_supported_scheme(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Legacy V1 string "base" is not CAIP-2; select_accept returns
        # None and the payer reports no_supported_scheme instead of
        # silently signing on the wrong network.
        body = _make_402_body()
        body["accepts"][0]["network"] = "base"

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json=body)

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/v1-only")
        assert result["error"] == "no_supported_scheme"

    def test_budget_gate_blocks_expensive_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Server wants $0.10; payer cap is $0.05. Must refuse before
        # creating a payment payload.
        requests: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            requests.append(req)
            return httpx.Response(402, json=_make_402_body(amount="100000"))

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/expensive")
        assert result["error"] == "budget_exceeded"
        assert result["reason"] == "max_per_call"
        # Exactly one upstream GET — the challenge. No paid retry.
        assert len(requests) == 1

    def test_ssrf_private_host_blocked(self) -> None:
        # No transport installed — the SSRF guard must refuse before any
        # httpx client is constructed.
        result = _payer().pay("http://127.0.0.1/foo")
        assert result["error"] == "unsafe_host"


class TestSubclassHooks:
    """Subclass hook wiring (the Xenarch commercial layer's contract).
    If this breaks, xenarch._payer.XenarchPayer stops working."""

    def test_pre_hook_short_circuits_before_budget_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        class _Gated(X402Payer):
            def _pre_payment_hook(
                self, *, url: str, accept: Any, price: Decimal
            ) -> dict[str, Any]:
                calls.append("pre")
                return {"error": "blocked_by_pre_hook"}

        def handler(req: httpx.Request) -> httpx.Response:
            if "x-payment" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            calls.append("paid-get")
            return httpx.Response(200, text="nope")

        _install_sync_transport(monkeypatch, handler)

        p = _Gated(
            private_key=Account.create().key.hex(),
            budget_policy=BudgetPolicy(),
            discover_via_pay_json=False,
        )
        result = p.pay("https://example.com/x")
        assert result["error"] == "blocked_by_pre_hook"
        # Paid GET never happened; session spend untouched.
        assert calls == ["pre"]
        assert p.budget_policy.session_spent == Decimal("0")

    def test_post_hook_mutates_success_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Tagger(X402Payer):
            def _post_payment_hook(
                self, result: dict[str, Any], paid_response: httpx.Response
            ) -> None:
                result["marker"] = "post-ran"

        def handler(req: httpx.Request) -> httpx.Response:
            if "x-payment" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="paid",
                headers={"X-PAYMENT-RESPONSE": _settle_header()},
            )

        _install_sync_transport(monkeypatch, handler)

        p = _Tagger(
            private_key=Account.create().key.hex(),
            budget_policy=BudgetPolicy(),
            discover_via_pay_json=False,
        )
        result = p.pay("https://example.com/x")
        assert result["success"] is True
        assert result["marker"] == "post-ran"


class TestAsyncHappyPath:
    async def test_async_pay_returns_success_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if "x-payment" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="async paid",
                headers={"X-PAYMENT-RESPONSE": _settle_header()},
            )

        _install_async_transport(monkeypatch, handler)

        result = await _payer().pay_async("https://example.com/async")
        assert result["success"] is True
        assert result["body"] == "async paid"
