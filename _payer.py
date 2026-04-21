"""Framework-agnostic x402 payer.

``X402Payer`` performs the full "GET → 402 → budget check → sign → retry →
return body" loop against any spec-compliant x402 v2 resource server. No
framework deps, no Xenarch branding, no facilitator coupling.

Framework adapters (LangChain, CrewAI, AutoGen, LangGraph) wrap this with
their tool-function contract. Xenarch's commercial layer subclasses it and
overrides ``_pre_payment_hook`` / ``_post_payment_hook`` to add signed
receipts and a reputation gate.

Both ``pay`` and ``pay_async`` return a plain ``dict``. Adapters decide how
to serialise (JSON string, tool-call response object, etc.). Neither raises
— every error is represented as ``{"error": "...", ...}`` in the result.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from x402.client import x402Client, x402ClientSync
from x402.mechanisms.evm.exact import register_exact_evm_client
from x402.mechanisms.evm.signers import EthAccountSigner
from x402.schemas import (
    PaymentRequired,
    PaymentRequirements,
    parse_payment_required,
)

from x402_agent._budget import BudgetPolicy
from x402_agent._helpers import (
    X_PAYMENT_HEADER,
    X_PAYMENT_RESPONSE_HEADER,
    budget_hint_exceeds,
    encode_payment_header,
    is_public_host,
    is_public_host_async,
    price_usd,
    select_accept,
    split_host_path,
    truncate_body,
)


class X402Payer:
    """Execute x402 payments. Framework-agnostic.

    One payer instance holds a pair of x402 SDK clients (sync + async) and a
    budget policy. Create one payer per agent if you need isolated session
    spend tracking.

    Subclass and override ``_pre_payment_hook`` / ``_post_payment_hook`` to
    add reputation gates, receipt fetches, metrics, etc. The hooks run
    inside the main pay loop so subclass errors surface as normal result
    dicts rather than raising.
    """

    def __init__(
        self,
        *,
        private_key: str,
        budget_policy: BudgetPolicy | None = None,
        discover_via_pay_json: bool = True,
        pay_json_timeout: float = 5.0,
        http_timeout: float = 10.0,
        max_response_bytes: int = 1_000_000,
    ) -> None:
        # Local import so ``eth_account`` is only required when a payer is
        # actually constructed — keeps the ``x402_agent`` import path cheap
        # for callers who only want the neutral helpers / BudgetPolicy.
        from eth_account import Account

        account = Account.from_key(private_key)
        signer = EthAccountSigner(account)

        self._signer_address = account.address
        # Sync + async clients are held separately: each is stateful w.r.t.
        # registered schemes and the async one binds internal work to the
        # running event loop.
        self._x402_sync = x402ClientSync()
        register_exact_evm_client(self._x402_sync, signer)
        self._x402_async = x402Client()
        register_exact_evm_client(self._x402_async, signer)

        self.budget_policy = budget_policy or BudgetPolicy()
        self.discover_via_pay_json = discover_via_pay_json
        self.pay_json_timeout = pay_json_timeout
        self.http_timeout = http_timeout
        self.max_response_bytes = max_response_bytes

    # ------------------------------------------------------------------
    # Hooks for subclasses. Default no-op; override to add policy.
    # ------------------------------------------------------------------

    def _pre_payment_hook(
        self,
        *,
        url: str,
        accept: PaymentRequirements,
        price: Decimal,
    ) -> dict[str, Any] | None:
        """Run before the budget lock. Return an error dict to abort."""
        return None

    async def _pre_payment_hook_async(
        self,
        *,
        url: str,
        accept: PaymentRequirements,
        price: Decimal,
    ) -> dict[str, Any] | None:
        return None

    def _post_payment_hook(
        self,
        result: dict[str, Any],
        paid_response: httpx.Response,
    ) -> None:
        """Mutate ``result`` after a successful paid GET. Never raises."""
        return None

    async def _post_payment_hook_async(
        self,
        result: dict[str, Any],
        paid_response: httpx.Response,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _parse_402(self, response: httpx.Response) -> PaymentRequired | None:
        """Return the parsed PaymentRequired (V2) or None if V1/invalid."""
        try:
            parsed = parse_payment_required(response.content)
        except (ValueError, TypeError, ValidationError, json.JSONDecodeError):
            return None
        # V1 still widely deployed but needs different scheme selection;
        # defer to a later release.
        if not isinstance(parsed, PaymentRequired):
            return None
        return parsed

    def _budget_gate(
        self,
        *,
        url: str,
        accept: PaymentRequirements,
        price: Decimal,
    ) -> dict[str, Any] | None:
        """Run the budget check + optional approval. Caller holds the lock."""
        budget_error = self.budget_policy.check(price)
        if budget_error is not None:
            return budget_error

        plan = {
            "url": url,
            "price_usd": str(price),
            "pay_to": accept.pay_to,
            "asset": accept.asset,
            "network": accept.network,
            "scheme": accept.scheme,
        }
        return self.budget_policy.request_approval(plan)

    def _success_response(
        self,
        *,
        url: str,
        response: httpx.Response,
        accept: PaymentRequirements,
        price: Decimal,
    ) -> dict[str, Any]:
        body = truncate_body(response.text, self.max_response_bytes)
        return {
            "success": True,
            "url": url,
            "amount_usd": str(price),
            "pay_to": accept.pay_to,
            "asset": accept.asset,
            "network": accept.network,
            "payment_response": response.headers.get(
                X_PAYMENT_RESPONSE_HEADER
            ),
            "body": body,
            "session_spent_usd": str(self.budget_policy.session_spent),
        }

    def _pay_json_pre_check(self, url: str) -> dict[str, Any] | None:
        """Fetch pay.json for the URL's host; return error dict or None."""
        if not self.discover_via_pay_json:
            return None

        # Local import: callers without the pay.json extra still load this
        # module without a hard import error.
        from pay_json import PayJson, PayJsonInvalid, PayJsonNotFound

        host, path = split_host_path(url)
        if not host:
            return None

        try:
            doc = PayJson.fetch(host, timeout=self.pay_json_timeout)
        except PayJsonNotFound:
            return None
        except PayJsonInvalid as exc:
            return {
                "error": "pay_json_invalid",
                "host": host,
                "details": str(exc),
            }
        except Exception:
            # Transport errors: fall through to the authoritative 402.
            return None

        rule = doc.match_rule(path)
        if rule is None or rule.budget_hints is None:
            return None

        return budget_hint_exceeds(rule.budget_hints, self.budget_policy)

    # ------------------------------------------------------------------
    # Public entry points.
    # ------------------------------------------------------------------

    def pay(self, url: str) -> dict[str, Any]:
        """Pay for ``url``; return a result dict. Never raises."""
        try:
            hostname = urlparse(url).hostname or ""
            if not is_public_host(hostname):
                return {"error": "unsafe_host", "host": hostname}

            pre_check = self._pay_json_pre_check(url)
            if pre_check is not None:
                return pre_check

            with httpx.Client(timeout=self.http_timeout) as client:
                initial = client.get(url)

                if initial.status_code != 402:
                    return {
                        "status": "no_payment_required",
                        "url": url,
                        "http_status": initial.status_code,
                        "body": truncate_body(
                            initial.text, self.max_response_bytes
                        ),
                    }

                payment_required = self._parse_402(initial)
                if payment_required is None:
                    return {
                        "error": "x402_parse_failed",
                        "url": url,
                        "body": truncate_body(initial.text, 500),
                    }

                accept = select_accept(payment_required)
                if accept is None:
                    return {
                        "error": "no_supported_scheme",
                        "url": url,
                        "accepts": [
                            {"scheme": a.scheme, "network": a.network}
                            for a in payment_required.accepts
                        ],
                    }

                price = price_usd(accept)

                # Pre-payment hook runs before the budget lock: a failing
                # subclass check (e.g. reputation lookup) must not block
                # other concurrent payments from acquiring the session
                # budget.
                pre_hook = self._pre_payment_hook(
                    url=url, accept=accept, price=price
                )
                if pre_hook is not None:
                    return pre_hook

                with self.budget_policy.lock():
                    gate_error = self._budget_gate(
                        url=url, accept=accept, price=price
                    )
                    if gate_error is not None:
                        return gate_error

                    payload = self._x402_sync.create_payment_payload(
                        payment_required
                    )
                    header_value = encode_payment_header(payload)

                    paid = client.get(
                        url, headers={X_PAYMENT_HEADER: header_value}
                    )

                    if paid.status_code != 200:
                        return {
                            "error": "x402_retry_failed",
                            "url": url,
                            "http_status": paid.status_code,
                            "body": truncate_body(paid.text, 500),
                        }

                    # Commit the moment the paid GET returns 200. Any
                    # later post-payment hook work must not revert this.
                    self.budget_policy.commit(price)

                    result = self._success_response(
                        url=url, response=paid, accept=accept, price=price
                    )
                    self._post_payment_hook(result, paid)
                    return result
        except httpx.HTTPError as exc:
            return {"error": "http_error", "kind": type(exc).__name__}
        except Exception as exc:  # noqa: BLE001 — payer contract is a dict
            return {"error": "unexpected_error", "kind": type(exc).__name__}

    async def pay_async(self, url: str) -> dict[str, Any]:
        try:
            hostname = urlparse(url).hostname or ""
            if not await is_public_host_async(hostname):
                return {"error": "unsafe_host", "host": hostname}

            # pay.json fetch is sync; offload so a slow host doesn't block
            # the event loop of an agent running many tools concurrently.
            pre_check = await asyncio.to_thread(self._pay_json_pre_check, url)
            if pre_check is not None:
                return pre_check

            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                initial = await client.get(url)

                if initial.status_code != 402:
                    return {
                        "status": "no_payment_required",
                        "url": url,
                        "http_status": initial.status_code,
                        "body": truncate_body(
                            initial.text, self.max_response_bytes
                        ),
                    }

                payment_required = self._parse_402(initial)
                if payment_required is None:
                    return {
                        "error": "x402_parse_failed",
                        "url": url,
                        "body": truncate_body(initial.text, 500),
                    }

                accept = select_accept(payment_required)
                if accept is None:
                    return {
                        "error": "no_supported_scheme",
                        "url": url,
                        "accepts": [
                            {"scheme": a.scheme, "network": a.network}
                            for a in payment_required.accepts
                        ],
                    }

                price = price_usd(accept)

                pre_hook = await self._pre_payment_hook_async(
                    url=url, accept=accept, price=price
                )
                if pre_hook is not None:
                    return pre_hook

                with self.budget_policy.lock():
                    gate_error = self._budget_gate(
                        url=url, accept=accept, price=price
                    )
                    if gate_error is not None:
                        return gate_error

                    payload = await self._x402_async.create_payment_payload(
                        payment_required
                    )
                    header_value = encode_payment_header(payload)

                    paid = await client.get(
                        url, headers={X_PAYMENT_HEADER: header_value}
                    )

                    if paid.status_code != 200:
                        return {
                            "error": "x402_retry_failed",
                            "url": url,
                            "http_status": paid.status_code,
                            "body": truncate_body(paid.text, 500),
                        }

                    self.budget_policy.commit(price)

                    result = self._success_response(
                        url=url, response=paid, accept=accept, price=price
                    )
                    await self._post_payment_hook_async(result, paid)
                    return result
        except httpx.HTTPError as exc:
            return {"error": "http_error", "kind": type(exc).__name__}
        except Exception as exc:  # noqa: BLE001
            return {"error": "unexpected_error", "kind": type(exc).__name__}
