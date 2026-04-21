# x402-agent

Framework-agnostic payer for the [x402 HTTP payment protocol](https://github.com/coinbase/x402).

Zero framework dependencies. Subclass or compose from **LangChain**, **CrewAI**, **AutoGen**, **LangGraph**, or anything else that wants to let an agent spend USDC against a `402 Payment Required` response without rolling its own pay loop.

## What it does

```python
from decimal import Decimal
from x402_agent import X402Payer, BudgetPolicy

payer = X402Payer(
    private_key="0x...",
    budget_policy=BudgetPolicy(
        max_per_call=Decimal("0.05"),
        max_per_session=Decimal("1.00"),
    ),
)

result = payer.pay("https://example.com/gated/article")
# {"success": True, "body": "...", "amount_usd": "0.01", ...}
```

`pay(url)` walks the full challenge: GET → parse `402` → pick a supported (scheme, network) → enforce your budget → sign an EIP-3009 USDC authorisation → retry with `X-PAYMENT` → return the unlocked body. Never raises; every failure mode is a dict you can show an LLM.

Async via `await payer.pay_async(url)`.

## What's in the package

| Symbol | Purpose |
| --- | --- |
| `X402Payer` | Neutral pay loop with `_pre_payment_hook` / `_post_payment_hook` for subclasses |
| `BudgetPolicy` | Per-call + per-session spend caps, optional human-approval callback, thread-safe |
| `select_accept` | Pick a supported `(scheme, network)` from a parsed `PaymentRequired` |
| `price_usd` | Atomic-units → `Decimal` USD using the advertised asset decimals |
| `is_public_host` / `is_public_host_async` | SSRF guard for agent-provided URLs |
| `budget_hint_exceeds` | Early-refusal check against `pay.json` `budget_hints` |

Constants (`DEFAULT_NETWORK`, `DEFAULT_SCHEME`, `X_PAYMENT_HEADER`, `X_PAYMENT_RESPONSE_HEADER`) are exported so adapters don't have to hard-code them.

## Subclassing

The four hooks are the intended extension point. Override any subset:

```python
class MyPayer(X402Payer):
    def _pre_payment_hook(self, *, url, accept, price):
        # Return an error dict to abort before the budget lock.
        if self._is_suspicious(accept.pay_to):
            return {"error": "refused_by_policy"}
        return None

    def _post_payment_hook(self, result, paid_response):
        # Mutate result in place; post runs after budget.commit().
        result["signed_by"] = "my-facilitator"
```

Pre-hook runs *before* the budget lock so a slow external check (reputation lookup, etc.) can't block other concurrent payments. Post-hook runs *after* the spend is committed so a receipt fetch failure can't revert a paid GET.

## Stability

`__all__` defines the v0.x stable surface. Anything in `x402_agent._helpers` or `x402_agent._payer` is accessible but subject to rename.

## Licence

MIT.
