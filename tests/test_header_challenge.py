"""Header-only V2 challenges. No wallets, signatures, network, or settlement."""
import base64
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from x402_agent._payer import X402Payer
from x402_agent._helpers import payment_headers, price_usd, select_accept

FIXTURE = json.loads(Path(__file__).with_name("header-challenge-fixture.json").read_text())


def payer():
    # Parsing happens before signing. Construct no wallet and attach no signer.
    return object.__new__(X402Payer)


def test_real_seller_header_only_challenge():
    response = httpx.Response(402, headers=FIXTURE["headers"], content=FIXTURE["body"])
    parsed = payer()._parse_402(response)
    assert parsed is not None
    assert parsed.x402_version == 2
    assert parsed.resource.url == FIXTURE["url"]
    accept = select_accept(parsed)
    assert accept.network == "eip155:84532"
    assert str(price_usd(accept)) == "0.001"
    assert payment_headers(parsed)[0].lower() == "payment-signature"


def test_header_precedes_conflicting_body():
    body = json.loads(base64.b64decode(FIXTURE["headers"]["PAYMENT-REQUIRED"]))
    body["accepts"][0]["amount"] = "999999"
    parsed = payer()._parse_402(httpx.Response(402, headers=FIXTURE["headers"], json=body))
    assert select_accept(parsed).amount == "1000"


@pytest.mark.parametrize("header", ["", "%%%", "e30=", "!!!!", "A" * 65537,
                                     base64.b64encode(b"\xff").decode(),
                                     base64.b64encode(b"not json").decode()])
def test_bad_header_never_falls_back_to_body(header):
    body = json.loads(base64.b64decode(FIXTURE["headers"]["PAYMENT-REQUIRED"]))
    assert payer()._parse_402(httpx.Response(402, headers={"PAYMENT-REQUIRED": header}, json=body)) is None


def test_body_v2_still_supported():
    body = base64.b64decode(FIXTURE["headers"]["PAYMENT-REQUIRED"])
    assert payer()._parse_402(httpx.Response(402, content=body)).x402_version == 2


def test_sync_loop_reaches_pre_payment_hook_without_signing(monkeypatch):
    p = payer()
    p.http_timeout = 1
    p.discover_via_pay_json = False
    p.max_response_bytes = 1000
    p._pre_payment_hook = Mock(return_value={"error": "fixture_stop_before_signing"})
    monkeypatch.setattr("x402_agent._payer.is_public_host", lambda _: True)
    real = httpx.Client
    transport = httpx.MockTransport(lambda _: httpx.Response(402, headers=FIXTURE["headers"], content="{}"))
    monkeypatch.setattr("x402_agent._payer.httpx.Client", lambda **kw: real(transport=transport, **kw))
    assert p.pay(FIXTURE["url"]) == {"error": "fixture_stop_before_signing"}
    p._pre_payment_hook.assert_called_once()


@pytest.mark.asyncio
async def test_async_loop_reaches_pre_payment_hook_without_signing(monkeypatch):
    p = payer()
    p.http_timeout = 1
    p.discover_via_pay_json = False
    p.max_response_bytes = 1000
    async def public(_): return True
    async def stop(**kw): return {"error": "fixture_stop_before_signing"}
    p._pre_payment_hook_async = stop
    monkeypatch.setattr("x402_agent._payer.is_public_host_async", public)
    real = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(402, headers=FIXTURE["headers"], content="{}"))
    monkeypatch.setattr("x402_agent._payer.httpx.AsyncClient", lambda **kw: real(transport=transport, **kw))
    assert await p.pay_async(FIXTURE["url"]) == {"error": "fixture_stop_before_signing"}
