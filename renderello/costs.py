"""Cost/latency economics (PDF acceptance gate): median latency, cost per
successful output, retries, and the gross-margin formula. All Magic Hour
figures are PROJECTIONS from published list data (docs.magichour.ai,
fetched 2026-11-14): model credit costs and subscription included-credit
rates. These catalog constants are not historical job receipts or measured latency.
Actual run evidence must be supplied separately. Effective $/credit uses each plan's included credits (a bulk
average, NOT a marginal price); credit-pack/usage-based rates vary by offer
and are not published numerically, so they are inputs, not defaults."""
from __future__ import annotations

import statistics
from dataclasses import dataclass


# docs.magichour.ai/api-reference/models (2026-11-14): "From" = credits for one
# image at the model's lowest resolution; larger resolutions cost more.
MODEL_CREDITS_FROM = {
    "flux-schnell": 5, "flux-2-klein": 5, "z-image-turbo": 5, "krea-2": 10,
    "qwen-edit": 10, "qwen-image-2.1": 10, "seedream-v4": 40,
    "seedream-v4.5": 50, "seedream-v5-flash": 30, "nano-banana": 50, "nano-banana-2-lite": 50,
    "nano-banana-2.1": 50, "gpt-image-2": 50, "seedream-v5-pro": 75,
    "nano-banana-2": 50, "gpt-image-2.5-flare": 100, "nano-banana-pro": 150,
    # 13 Nov 2026 settled receipts: 200/image at res=1k, 1:1, single input.
    "gpt-image-2.5-flare-1k": 200,
}

# docs.magichour.ai/billing/overview (2026-11-14): subscription included credits.
# effective_usd_per_credit = monthly price / monthly credits (bulk included rate).
SUBSCRIPTION_PLANS = {
    "creator_monthly": {"usd": 19, "credits": 12_000},
    "creator_annual": {"usd": 12, "credits": 12_000},
    "pro_monthly": {"usd": 39, "credits": 25_000},
    "pro_annual": {"usd": 25, "credits": 25_000},
    "business_monthly": {"usd": 99, "credits": 70_000},
    "business_annual": {"usd": 66, "credits": 70_000},
}


@dataclass(frozen=True)
class RouteEconomics:
    route: str
    runs: int
    successes: int
    retries: int                       # runs beyond the first for a success
    median_seconds: float | None       # measured, real runs only
    credits_total: float               # recorded on the runs (0 for self-hosted)
    credits_per_successful_output: float | None


def route_economics(route: str, rows: list[dict]) -> RouteEconomics:
    """rows: benchmark/result dicts with seconds, credits_estimated, success."""
    times = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    successes = sum(1 for r in rows if r.get("success", True))
    credits = sum(float(r.get("credits_estimated", 0)) for r in rows)
    return RouteEconomics(
        route=route, runs=len(rows), successes=successes,
        retries=max(0, len(rows) - successes) if successes else len(rows),
        median_seconds=round(statistics.median(times), 2) if times else None,
        credits_total=credits,
        credits_per_successful_output=round(credits / successes, 2) if credits and successes else None)


def effective_usd_per_credit(plan: str) -> float:
    """Bulk included-credit rate for a subscription plan. NOT a marginal price."""
    p = SUBSCRIPTION_PLANS[plan]
    return p["usd"] / p["credits"]


def projected_usd_per_output(model: str, plan: str, images_per_generation: int = 1,
                             credits_per_generation: float | None = None) -> float:
    """Projected $ cost of one MH generation under a plan's included-credit rate.
    credits_per_generation overrides the published 'from' rate (e.g. higher
    resolution or multi-image batches cost more than the single-image floor)."""
    credits = credits_per_generation if credits_per_generation is not None \
        else MODEL_CREDITS_FROM[model] * images_per_generation
    return round(credits * effective_usd_per_credit(plan), 6)


def gross_margin_per_output(sale_price_usd: float, cost_usd: float) -> float:
    """Margin formula only: the sale price is a caller input, never assumed."""
    return round(sale_price_usd - cost_usd, 6)
