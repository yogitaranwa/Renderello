"""S2 comparison runner: run the SAME source + prompt through 2-3 Magic Hour
image-editor models and record outputs, latency and credits side by side.

Two layers, kept strictly separate:
- plan_compare(): pure projection from the published table. Zero network, zero
  spend. This is the default CLI behavior.
- run_compare(): executes one edit per model. With live backends this SPENDS
  Magic Hour credits and therefore requires explicit --live on the CLI (owner
  approval). Tests drive it with stub backends (zero network, zero spend).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .backends.magichour import MODEL_TABLE, MagicHourBackend
from .costs import effective_usd_per_credit
from .prompts import build_negative_prompt, build_prompt
from .schemas import Aspect, EditRequest, TemplateSpec

DEFAULT_COMPARE_SET = ["flux-2-klein", "qwen-edit", "seedream-v4"]


def plan_compare(models: list[str], plan: str = "creator_monthly") -> dict:
    """Zero-spend projection of what a live comparison would do and cost."""
    rows = []
    for m in models:
        if m not in MODEL_TABLE:
            raise KeyError(f"unknown model '{m}'. known: {', '.join(sorted(MODEL_TABLE))}")
        t = MODEL_TABLE[m]
        rows.append({
            "model": m,
            "credits_listed_from": t["credits"],
            "tier_required": t["tier"],
            "max_inputs_editor": t["max_inputs"],
            "projected_usd_per_image": round(t["credits"] * effective_usd_per_credit(plan), 6),
        })
    return {
        "kind": "plan_only_no_spend",
        "plan_basis": plan,
        "note": "projections use published 'From' single-image lowest-res rates "
                "and the plan's bulk included-credit rate; not marginal cost, "
                "not measured latency",
        "models": rows,
        "total_credits_listed": sum(r["credits_listed_from"] for r in rows),
        "total_projected_usd": round(sum(r["projected_usd_per_image"] for r in rows), 6),
    }


def run_compare(source_path: Path, models: list[str], template: TemplateSpec,
                seed: int, aspect: Aspect, out_dir: Path,
                backend_factory=None, live: bool = False) -> dict:
    """One edit per model, same source/prompt/aspect; seed local/mock only. Returns the manifest
    and writes it to out_dir/compare_manifest.json plus one PNG per model."""
    if backend_factory is None:
        def backend_factory(model: str):
            return MagicHourBackend(model=model, live=live,
                                    cost_log=out_dir / "cost_log.jsonl",
                                    receipt_path=out_dir / "project_receipts.jsonl")
    plan_compare(models)  # validate every model before any network/spend
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(source_path).convert("RGB")
    req = EditRequest(source_path=source_path, template=template, seed=seed)
    prompt = build_prompt(req)
    negative = build_negative_prompt()
    manifest_path = out_dir / "compare_manifest.json"
    if manifest_path.exists() or (out_dir / "project_receipts.jsonl").exists():
        raise FileExistsError("Comparison evidence already exists. Recover recorded project IDs; "
                              "never repeat a possibly charged run in this directory.")
    manifest = {
        "kind": "mh_model_comparison", "state": "running",
        "source": str(source_path), "models_requested": models,
        "template": template.key, "seed": seed, "aspect": aspect.value,
        "seed_note": "local/mock seed only; MH API does not accept a seed",
        "negative_prompt_note": "recorded only; MH API has no negative_prompt field",
        "prompt": prompt, "negative_prompt": negative,
        "note": "listed credits are price floors, not measured charges; missing actual charge is unknown",
        "results": [],
    }
    def checkpoint():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(manifest_path)
    checkpoint()  # directory is claimed BEFORE the first possibly charged call
    for m in models:
        backend = backend_factory(m)
        row = {"model": m, "live": getattr(backend, "live", False),
               "state": "submitting", "credits_listed_from": MODEL_TABLE[m]["credits"]}
        manifest["results"].append(row)
        checkpoint()
        t0 = time.monotonic()
        try:
            img = backend.edit(source, prompt, negative, seed, aspect)
            out_path = out_dir / f"compare_{m}_seed{seed}.png"
            img.save(out_path)
            row.update(state="complete", output=str(out_path))
        except Exception as exc:
            row.update(state="failed_or_unconfirmed", error=str(exc),
                       recovery="Check receipts/provider projects. Do not resubmit automatically.")
            manifest["state"] = "interrupted"
            raise
        finally:
            result = getattr(backend, "last_result", None) or {}
            row.update(latency_s_measured=round(time.monotonic()-t0,3),
                       project_id=result.get("id"),
                       credits_charged=(result.get("final") or {}).get("credits_charged"),
                       credits_at_submission=(result.get("submission") or {}).get("credits_charged"))
            checkpoint()
    manifest["state"] = "complete"
    checkpoint()
    return manifest
