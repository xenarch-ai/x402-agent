"""Pure helpers for x402 payment flows.

Framework-agnostic primitives shared by every agent-framework adapter and by
Xenarch's commercial layer. No httpx, no pydantic BaseModel, no framework
imports. Only standard library + the ``x402`` SDK types.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from x402.http.constants import (
    PAYMENT_RESPONSE_HEADER as PAYMENT_RESPONSE_HEADER,
    PAYMENT_SIGNATURE_HEADER as PAYMENT_SIGNATURE_HEADER,
    X_PAYMENT_HEADER as X_PAYMENT_HEADER,
    X_PAYMENT_RESPONSE_HEADER as X_PAYMENT_RESPONSE_HEADER,
)
from x402.mechanisms.evm.constants import DEFAULT_DECIMALS
from x402.mechanisms.evm.v1.constants import V1_NETWORKS
from x402.schemas import (
    PaymentRequired,
    PaymentRequiredV1,
    PaymentRequirements,
    PaymentRequirementsV1,
)


# The x402 server advertises the atomic amount as an integer string in the
# asset's smallest unit; for USDC that is 6 decimals. The authoritative
# value can live in ``requirements.extra["decimals"]`` — we honour that
# when present and fall back to the EVM-default (6) otherwise. V1 names
# the field ``max_amount_required``; V2 renamed it to ``amount``.
#
# Header constants are re-exported from the upstream SDK so a single
# rename in ``x402.http.constants`` propagates here. V1 wire uses
# ``X-PAYMENT`` / ``X-PAYMENT-RESPONSE``; V2 dropped the ``X-`` prefix
# (HTTP RFC 6648) and uses ``PAYMENT-SIGNATURE`` / ``PAYMENT-RESPONSE``.

# Default preferred network — CAIP-2 chain ID for Base (8453). The x402 v2
# spec identifies networks with CAIP-2 identifiers and the SDK registers
# an ``eip155:*`` wildcard, so any EVM chain with that prefix is payable.
# V1 uses legacy name strings; ``"base"`` is the Base-mainnet legacy name
# and the V1 preferred fallback.
DEFAULT_NETWORK = "eip155:8453"
DEFAULT_NETWORK_V1 = "base"
DEFAULT_SCHEME = "exact"
EIP155_PREFIX = "eip155:"
V1_NETWORKS_SET = frozenset(V1_NETWORKS)

AnyPaymentRequired = PaymentRequired | PaymentRequiredV1
AnyPaymentRequirements = PaymentRequirements | PaymentRequirementsV1


def payment_headers(
    payment_required: AnyPaymentRequired,
) -> tuple[str, str]:
    """Return ``(request_header, response_header)`` for the wire version.

    V1 gates expect ``X-PAYMENT`` on the retry and emit
    ``X-PAYMENT-RESPONSE`` on the settled 200. V2 gates expect
    ``PAYMENT-SIGNATURE`` and emit ``PAYMENT-RESPONSE`` (the ``X-``
    prefix was dropped per RFC 6648). Sending the V1 header to a V2
    facilitator is what caused PayAI to ignore signed retries with a
    fresh 402 in Phase A.
    """
    if isinstance(payment_required, PaymentRequiredV1):
        return X_PAYMENT_HEADER, X_PAYMENT_RESPONSE_HEADER
    return PAYMENT_SIGNATURE_HEADER, PAYMENT_RESPONSE_HEADER


def _atomic_amount(req: AnyPaymentRequirements) -> Decimal:
    """Return the atomic on-chain amount from either a V2 or V1 entry."""
    if isinstance(req, PaymentRequirementsV1):
        return Decimal(req.max_amount_required)
    return Decimal(req.amount)


def price_usd(req: AnyPaymentRequirements) -> Decimal:
    """Convert atomic on-chain amount to Decimal USD using asset decimals.

    Accepts both V1 (``max_amount_required``) and V2 (``amount``) shapes.
    """
    decimals = DEFAULT_DECIMALS
    extra = req.extra or {}
    extra_decimals = extra.get("decimals")
    if isinstance(extra_decimals, int) and extra_decimals >= 0:
        decimals = extra_decimals
    amount = _atomic_amount(req)
    scale = Decimal(10) ** decimals
    return amount / scale


def select_accept(
    payment_required: AnyPaymentRequired,
    *,
    preferred_scheme: str = DEFAULT_SCHEME,
    preferred_network: str = DEFAULT_NETWORK,
    preferred_network_v1: str = DEFAULT_NETWORK_V1,
) -> AnyPaymentRequirements | None:
    """Pick the first accept entry that matches our registered scheme/network.

    Handles both V1 and V2 payment-required responses. The SDK's EVM client
    registers schemes on the V2 ``eip155:*`` wildcard and on every legacy
    V1 network name, so any of these can settle.

    Preference order:
      1. V2 exact match on (scheme, preferred_network).
      2. V2 same scheme on any CAIP-2 ``eip155:`` chain.
      3. V1 exact match on (scheme, preferred_network_v1), e.g. ``"base"``.
      4. V1 same scheme on any registered legacy EVM network.
      5. Give up.
    """
    accepts = payment_required.accepts
    # V2 preferences first — modern servers emit V2.
    for entry in accepts:
        if isinstance(entry, PaymentRequirements):
            if entry.scheme == preferred_scheme and entry.network == preferred_network:
                return entry
    for entry in accepts:
        if isinstance(entry, PaymentRequirements):
            if (
                entry.scheme == preferred_scheme
                and entry.network.startswith(EIP155_PREFIX)
            ):
                return entry
    # V1 fallbacks — many production gates still emit V1.
    for entry in accepts:
        if isinstance(entry, PaymentRequirementsV1):
            if (
                entry.scheme == preferred_scheme
                and entry.network == preferred_network_v1
            ):
                return entry
    for entry in accepts:
        if isinstance(entry, PaymentRequirementsV1):
            if (
                entry.scheme == preferred_scheme
                and entry.network in V1_NETWORKS_SET
            ):
                return entry
    return None


def encode_payment_header(payload: Any) -> str:
    """Base64-encode the JSON form of a x402 payment payload for X-PAYMENT."""
    return base64.b64encode(
        payload.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    ).decode("ascii")


def truncate_body(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def split_host_path(url: str) -> tuple[str, str]:
    """Split a URL into (host, path) for pay.json resolution.

    Returns the host as ``hostname[:port]`` with any userinfo stripped,
    so error echoes and pay.json fetches never leak credentials embedded
    in a URL like ``https://user:pass@example.com/foo``. Path falls back
    to ``"/"`` when empty so the rule matcher always has something to
    glob against — ``match_rule("")`` is ill-defined.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    host = f"{hostname}:{parsed.port}" if parsed.port else hostname
    path = parsed.path or "/"
    return host, path


def is_public_host(host: str) -> bool:
    """True iff ``host`` resolves only to globally-routable IP addresses.

    Blocks SSRF to loopback, RFC1918 private ranges, link-local (including
    AWS/GCP IMDS at 169.254.169.254), multicast, and reserved/unspecified
    space. An agent-provided URL like ``http://169.254.169.254/latest``
    would otherwise let a prompt-injection attack read cloud metadata into
    the LLM's context.

    Best-effort only: a TOCTOU window exists between this resolve and the
    actual connect, and DNS rebinding can defeat it. Treat as defence-in-
    depth on top of network-level egress rules.
    """
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        ):
            return False
    return True


async def is_public_host_async(host: str) -> bool:
    """Async variant of ``is_public_host`` that doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, is_public_host, host)


def budget_hint_exceeds(
    rule_hints: dict[str, Any],
    policy: Any,
) -> dict[str, Any] | None:
    """Return an error dict if rule-advertised caps exceed the local policy.

    We compare on two knobs: per-call and per-session. The publisher's
    hint is advisory (``recommended_max_per_call``), not authoritative —
    we use it only to short-circuit hopeless fetches. When hints are
    malformed or missing, treat as "no guidance" and fall through to the
    live 402 price check.

    ``policy`` is typed as ``Any`` so this helper is usable against any
    budget-like object with ``max_per_call`` / ``max_per_session``
    attributes. Callers in practice pass a ``BudgetPolicy``.
    """
    per_call_hint = rule_hints.get("recommended_max_per_call")
    per_session_hint = rule_hints.get("recommended_max_per_session")

    def _as_decimal(raw: Any) -> Decimal | None:
        if not isinstance(raw, str):
            return None
        try:
            value = Decimal(raw)
        except Exception:
            return None
        return value if value.is_finite() and value >= 0 else None

    per_call = _as_decimal(per_call_hint)
    if per_call is not None and per_call > policy.max_per_call:
        return {
            "error": "budget_hint_exceeded",
            "reason": "recommended_max_per_call",
            "hint_usd": str(per_call),
            "limit_usd": str(policy.max_per_call),
        }

    per_session = _as_decimal(per_session_hint)
    if per_session is not None and per_session > policy.max_per_session:
        return {
            "error": "budget_hint_exceeded",
            "reason": "recommended_max_per_session",
            "hint_usd": str(per_session),
            "limit_usd": str(policy.max_per_session),
        }

    return None
