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

from x402.mechanisms.evm.constants import DEFAULT_DECIMALS
from x402.schemas import PaymentRequired, PaymentRequirements


# The x402 server advertises ``amount`` as an integer string in the asset's
# smallest unit; for USDC that is 6 decimals. The authoritative value can
# live in ``requirements.extra["decimals"]`` — we honour that when present
# and fall back to the EVM-default (6) otherwise.
X_PAYMENT_HEADER = "X-PAYMENT"
X_PAYMENT_RESPONSE_HEADER = "X-PAYMENT-RESPONSE"

# Default preferred network — CAIP-2 chain ID for Base (8453). The x402 v2
# spec identifies networks with CAIP-2 identifiers and the SDK registers an
# ``eip155:*`` wildcard, so any EVM chain with that prefix is payable. We
# prefer Base explicitly but fall back to any CAIP-2 eip155 entry so a
# resource server advertising ``eip155:1`` (Ethereum) still works out of
# the box. Non-CAIP-2 legacy strings like ``"base"`` are V1 and not handled.
DEFAULT_NETWORK = "eip155:8453"
DEFAULT_SCHEME = "exact"
EIP155_PREFIX = "eip155:"


def price_usd(req: PaymentRequirements) -> Decimal:
    """Convert atomic on-chain amount to Decimal USD using asset decimals."""
    decimals = DEFAULT_DECIMALS
    extra = req.extra or {}
    extra_decimals = extra.get("decimals")
    if isinstance(extra_decimals, int) and extra_decimals >= 0:
        decimals = extra_decimals
    amount = Decimal(req.amount)
    scale = Decimal(10) ** decimals
    return amount / scale


def select_accept(
    payment_required: PaymentRequired,
    *,
    preferred_scheme: str = DEFAULT_SCHEME,
    preferred_network: str = DEFAULT_NETWORK,
) -> PaymentRequirements | None:
    """Pick the first accept entry that matches our registered scheme/network.

    Preference order:
      1. Exact match on (scheme, network) — e.g. (exact, eip155:8453).
      2. Same scheme on any CAIP-2 ``eip155:`` chain — the SDK's V2 EVM
         client is registered under ``eip155:*`` so any EVM network works.
      3. Give up. V1 legacy networks (e.g. plain ``"base"``) are rejected;
         the caller gets a ``no_supported_scheme`` error upstream.
    """
    for entry in payment_required.accepts:
        if entry.scheme == preferred_scheme and entry.network == preferred_network:
            return entry
    for entry in payment_required.accepts:
        if (
            entry.scheme == preferred_scheme
            and entry.network.startswith(EIP155_PREFIX)
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
