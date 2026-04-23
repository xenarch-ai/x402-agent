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


def _make_402_body_v1(amount: str = "10000") -> dict[str, Any]:
    # Wire format emitted by a V1 gate: flat ``resource``, camel-case
    # ``maxAmountRequired``, legacy network name ``"base"``. This matches
    # what gate.xenarch.dev and other production WP gates return today.
    return {
        "x402Version": 1,
        "error": "payment_required",
        "accepts": [
            {
                "scheme": "exact",
                "network": "base",
                "maxAmountRequired": amount,
                "resource": "https://example.com/article/1",
                "description": "Paid content",
                "mimeType": "text/html",
                "payTo": "0x0000000000000000000000000000000000000001",
                "maxTimeoutSeconds": 60,
                "asset": USDC,
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
            # V2 retry header is ``PAYMENT-SIGNATURE``; V1 ``X-PAYMENT``
            # would never arrive here because the body advertises V2.
            if "payment-signature" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="paid body",
                headers={"PAYMENT-RESPONSE": _settle_header()},
            )

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/article/1")
        assert result["success"] is True
        assert result["body"] == "paid body"
        assert result["amount_usd"] == "0.01"
        # Session budget must advance by exactly the paid amount.
        assert result["session_spent_usd"] == "0.01"
        # Settlement header surfaces in result regardless of wire version.
        assert result["payment_response"] is not None

    def test_v2_retry_uses_payment_signature_not_x_payment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin the regression: a V2 facilitator that sees ``X-PAYMENT``
        # treats it as a missing signature and re-issues 402. This test
        # fails on x402-agent <=0.1.2 which hardcoded ``X-PAYMENT``.
        retry_headers: list[set[str]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            lowered = {k.lower() for k in req.headers}
            if "payment-signature" in lowered:
                retry_headers.append(lowered)
                return httpx.Response(
                    200,
                    text="ok",
                    headers={"PAYMENT-RESPONSE": _settle_header()},
                )
            if "x-payment" in lowered:
                # Wrong header — fail loudly so the test catches it.
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(402, json=_make_402_body())

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/v2-article")
        assert result["success"] is True, result
        assert len(retry_headers) == 1
        assert "payment-signature" in retry_headers[0]
        assert "x-payment" not in retry_headers[0]

    def test_v1_402_challenge_then_paid_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Production WP gates (and any pre-V2 spec-compliant server) emit
        # ``x402Version: 1`` with ``maxAmountRequired`` and legacy network
        # names. The SDK's EVM client registers V1 for all legacy chains,
        # so this must settle end-to-end.
        def handler(req: httpx.Request) -> httpx.Response:
            if "x-payment" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body_v1())
            return httpx.Response(
                200,
                text="v1 paid body",
                headers={
                    "X-PAYMENT-RESPONSE": base64.b64encode(
                        json.dumps(
                            {
                                "success": True,
                                "transaction": "0xcafebabe",
                                "network": "base",
                            }
                        ).encode()
                    ).decode()
                },
            )

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/v1-article")
        assert result["success"] is True, result
        assert result["body"] == "v1 paid body"
        assert result["amount_usd"] == "0.01"
        assert result["network"] == "base"

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
        # V1 advertising a legacy network the SDK hasn't registered
        # (e.g. "solana"). select_accept returns None and the payer
        # reports no_supported_scheme instead of signing the wrong chain.
        body = _make_402_body_v1()
        body["accepts"][0]["network"] = "solana"

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json=body)

        _install_sync_transport(monkeypatch, handler)

        result = _payer().pay("https://example.com/unsupported")
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
            if "payment-signature" not in {k.lower() for k in req.headers}:
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
            if "payment-signature" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="paid",
                headers={"PAYMENT-RESPONSE": _settle_header()},
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


class TestPayJsonSoftImport:
    """pay.json discovery must be optional — no crash without the extra.

    The ``pay-json`` extra is genuinely optional. A user can ``pip install
    x402-agent`` with no extras, pass ``discover_via_pay_json=True`` (the
    default), and the payer must still work: skip discovery, go straight
    to the 402, settle normally. A prior version hard-imported ``pay_json``
    at function scope and crashed every call with ``unexpected_error``
    when the extra wasn't installed. This regression test pins that fix.
    """

    def test_missing_pay_json_module_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _blocking_import(
            name: str, globals: Any = None, locals: Any = None,
            fromlist: Any = (), level: int = 0,
        ) -> Any:
            if name == "pay_json" or name.startswith("pay_json."):
                raise ImportError(
                    "No module named 'pay_json' (simulated by test)"
                )
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        def handler(req: httpx.Request) -> httpx.Response:
            if "payment-signature" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="paid",
                headers={"PAYMENT-RESPONSE": _settle_header()},
            )

        _install_sync_transport(monkeypatch, handler)

        # discover_via_pay_json=True is the default; set it explicitly to
        # document that this code path is the one we're guarding.
        p = X402Payer(
            private_key=Account.create().key.hex(),
            budget_policy=BudgetPolicy(),
            discover_via_pay_json=True,
        )
        result = p.pay("https://example.com/x")
        assert result["success"] is True, result


class TestAsyncHappyPath:
    async def test_async_pay_returns_success_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if "payment-signature" not in {k.lower() for k in req.headers}:
                return httpx.Response(402, json=_make_402_body())
            return httpx.Response(
                200,
                text="async paid",
                headers={"PAYMENT-RESPONSE": _settle_header()},
            )

        _install_async_transport(monkeypatch, handler)

        result = await _payer().pay_async("https://example.com/async")
        assert result["success"] is True
        assert result["body"] == "async paid"

    async def test_async_v2_retry_uses_payment_signature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retry_headers: list[set[str]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            lowered = {k.lower() for k in req.headers}
            if "payment-signature" in lowered:
                retry_headers.append(lowered)
                return httpx.Response(
                    200,
                    text="ok",
                    headers={"PAYMENT-RESPONSE": _settle_header()},
                )
            return httpx.Response(402, json=_make_402_body())

        _install_async_transport(monkeypatch, handler)

        result = await _payer().pay_async("https://example.com/async-v2")
        assert result["success"] is True, result
        assert len(retry_headers) == 1
        assert "x-payment" not in retry_headers[0]
