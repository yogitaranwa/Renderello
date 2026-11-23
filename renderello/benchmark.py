"""Benchmark harness. Consumes a manifest of 20+ source images (the plan's
acceptance gate: varied sources incl. tiny-text skincare bottle, reflective can,
transparent glass, shoes, bags, apparel, food packaging, multi-product, complex
edges, irregular shapes), runs the full pipeline per source x template, and
aggregates QA into scores.csv + a per-run report.

Manifest format (benchmark/manifest.json):
{
  "sources": [{"path": "benchmark/sources/skincare_bottle.jpg",
               "label": "skincare bottle with tiny text",
               "license": "own photo | CC0 | ...",   // recorded per plan
               "notes": "optional"}],
  "templates": ["clean_studio", "luxury_hero"],        // subset or "all"
  "routes": ["cutout_compose", "pure_prompt", "hybrid"],
  "variants": 2
}
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image  # noqa: F401 - validates sources up front

from .pipeline import run_pipeline
from .qa import report_to_csv_row
from .config import RunConfig
from .schemas import EditRequest, Route
from .templates import all_templates, get_template


def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text())
    if len(m.get("sources", [])) < 1:
        raise ValueError("manifest has no sources")
    for s in m["sources"]:
        if not (path.parent / s["path"]).exists():
            raise FileNotFoundError(f"source missing: {s['path']}")
        if "license" not in s:
            raise ValueError(f"source {s['path']} lacks a license record (plan B6)")
    from .source_contract import SourceContract
    for source in m['sources']:
        SourceContract(**source.get('source_contract', {}))
        if 'templates' in source:
            if not source['templates']:raise ValueError('source template list is empty')
            for key in source['templates']:get_template(key)
    if 'variants' in m and (type(m['variants']) is not int or not 1<=m['variants']<=4):
        raise ValueError('manifest variants must be an integer 1..4')
    return m


def run_benchmark(manifest_path: Path, cfg: RunConfig,
                  editor=None, out_dir: Path | None = None,
                  scene_fn=None, scene_kind: str = "", allow_colour_conversion: bool = False,
                  scene_finish: bool = False, scene_accept=None, scene_max_attempts: int = 1) -> Path:
    m = load_manifest(manifest_path)
    templates = all_templates() if m.get("templates") == "all" else \
        [get_template(k) for k in m["templates"]]
    routes = [Route(r) for r in m["routes"]]
    out = out_dir or cfg.out_root / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    # Manifest declares repeat counts and reviewer inputs, never permission for
    # external generation or a quality claim. Invalid declarations fail before
    # invoking any source/model, rather than losing failures from the denominator.
    from .source_contract import SourceContract
    cfg=replace(cfg,variants=m.get('variants',cfg.variants))
    # Validate the whole source list before any generation/model work.
    # A late ICC/depth/orientation error must not appear only after prior calls.
    from .pipeline_intake import load_generation_image
    for src in m["sources"]:
        probe, _ = load_generation_image(manifest_path.parent / src["path"],
                                        allow_colour_conversion=allow_colour_conversion)
        probe.close()
    rows: list[dict] = []
    runs = 0
    for src in m["sources"]:
        src_path = manifest_path.parent / src["path"]
        source_templates=([get_template(k) for k in src['templates']]
                          if 'templates' in src else templates)
        contract=SourceContract(**src.get('source_contract',{}))
        for tpl in source_templates:
            req = EditRequest(source_path=src_path, template=tpl,
                              custom_instruction=src.get("instruction", ""),
                              seed=cfg.seed)
            result = run_pipeline(req, routes, cfg, editor=editor, scene_fn=scene_fn,
                                  glass_mode=bool(src.get("glass", False)), source_contract=contract,
                                  require_complete=bool(src.get('require_complete',False)),
                                  allow_colour_conversion=allow_colour_conversion,
                                  scene_finish=scene_finish,
                                  scene_accept=scene_accept, scene_max_attempts=scene_max_attempts)
            runs += 1
            source_record = json.loads(result.manifest_path.read_text()).get("source_provenance", {})
            for v in result.variants:
                row = report_to_csv_row(v.qa)
                row.update({"source": str(src_path.resolve()),
                            "source_label": src["label"],
                            "source_sha256": source_record.get("sha256"),
                            "source_classification": source_record.get("classification", "UNKNOWN / unclassified"),
                            "source_match_basis": source_record.get("match_basis", "unrecorded"),
                            "source_license": src["license"],
                            "backend": v.backend,
                            "seconds": v.seconds,
                            "credits_estimated": v.credits_estimated,
                            "success": v.success and v.output_path.is_file(),
                            "scene_kind": scene_kind if v.route is Route.SCENE_COMPOSE else "",
                            "source_contract": json.dumps(asdict(contract),sort_keys=True),
                            "require_complete": bool(src.get('require_complete',False))})
                rows.append(row)

    csv_path = out / "scores.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Gate view per the problem statement, with explicit numerators/denominators:
    # - >=80% of outputs with no critical product alteration
    # - >=70% publish-ready (fidelity 4-5 AND no critical flags; rubric 1-5)
    # - >=8 templates represented in the run
    # - config comparison: best route vs cheaper fallback by mean fidelity
    # - every critical-flagged variant lands in the human-review queue
    summary = _summarize(rows, runs)
    if scene_kind:
        summary["scene_kind"] = scene_kind + (" (MOCK gradient: plumbing only, not scene quality)" if scene_kind == "mock" else "")
    (out / "benchmark_report.json").write_text(json.dumps(summary, indent=2))
    return csv_path


def _summarize(rows: list[dict], runs: int, human: dict | None = None) -> dict:
    """Shared gate math for fresh benchmark runs and summarize_runs - one
    implementation so the two paths can never drift apart."""
    summary: dict = {"runs": runs, "variants": len(rows), "routes": {},
                     "templates_represented": sorted({r["template"] for r in rows}),
                     "gate_status": "NOT EVALUATED on real data - these numbers "
                     "come from whatever sources the manifest held; the plan's "
                     "targets are achieved only on the full 20+ real-source run "
                     "with human eye review of flagged outputs",
                     "quality_status": "NOT_EVALUATED"}
    known_sources = {r["source"] for r in rows if r.get("source")}
    source_hashes = {r["source_sha256"] for r in rows if r.get("source_sha256")}
    content_complete = bool(rows) and all(bool(r.get("source_sha256")) for r in rows)
    classes = {}
    for row in rows:
        kind = row.get("source_classification") or "UNKNOWN / unclassified"
        identity = row.get("source_sha256") or row.get("source") or "unrecorded"
        classes.setdefault(kind, set()).add(identity)
    summary["coverage"] = {
        "unique_sources": len(known_sources), "source_target": 20,
        "unique_content_sources": len(source_hashes),
        "source_identity_complete": content_complete,
        "source_count_met": content_complete and len(source_hashes) >= 20,
        "source_count_basis": "distinct recorded file SHA256, not paths or verified photo capture origin; different encodings of one image can still count separately",
        "source_classifications": {k: len(v) for k, v in sorted(classes.items())},
        "real_photo_count_met": None,
        "real_photo_status": "NOT_VERIFIED: photographic appearance does not establish capture origin. Illustrations and unknown inputs must not be called verified photographs.",
        "templates": len(summary["templates_represented"]), "template_target": 8,
        "template_count_met": len(summary["templates_represented"]) >= 8,
    }
    # Scope and specialist evidence are independent of per-sample ratings.
    summary["acceptance_status"] = "NOT_EVALUATED"
    summary["acceptance_pass"] = None
    summary["specialist_comparison_status"] = "NOT_EVALUATED"
    n = len(rows)
    if n:
        no_crit = sum(1 for r in rows if r.get("success", True) and not r["critical_errors"])
        publish = sum(1 for r in rows
                      if r.get("success", True) and not r["critical_errors"]
                      and (r.get("score_fidelity") or 0) >= 4)
        summary["gates"] = {
            "no_critical_alteration": {"numerator": no_crit, "denominator": n,
                                       "rate": round(no_crit / n, 3),
                                       "target": 0.80,
                                       "pass": None,
                                       "status": "NOT_EVALUATED",
                                       "automated_proxy_pass": (no_crit / n) >= 0.80,
                                       "basis": "automated tripwires only; a human "
                                                "eye audit can only LOWER this rate "
                                                "(see README known QA gap)"},
            "publish_ready": {"numerator": publish, "denominator": n,
                              "rate": round(publish / n, 3), "target": 0.70,
                              "pass": None,
                              "status": "NOT_EVALUATED",
                              "automated_proxy_pass": (publish / n) >= 0.70,
                              "basis": "STRUCTURAL PROXY (auto fidelity>=4 and no "
                                       "flags) - NOT a human publishability rating; "
                                       "the PDF gate needs human-scored 1-5 "
                                       "publishability on real outputs"},
            "templates_represented": {
                "numerator": len(summary["templates_represented"]),
                "denominator": 8,
                "pass": len(summary["templates_represented"]) >= 8},
            "human_review_queue": [
                {"output": r["output"], "template": r["template"],
                 "flags": r["critical_errors"],
                 "reason": ("GENERATIVE_PRODUCT_EDIT_ALWAYS_REQUIRES_FIDELITY_REVIEW"
                            if r.get("route") in ("pure_prompt", "hybrid") else "AUTOMATED_WARNING_REQUIRES_REVIEW")}
                for r in rows if r.get("success", True) and (r["critical_errors"] or r.get("route") in ("pure_prompt", "hybrid"))],
            "execution_failure_queue": [
                {"output": r["output"], "template": r["template"], "errors": r["critical_errors"],
                 "status": "FAILED_NOT_A_QUALITY_REVIEW_CANDIDATE"}
                for r in rows if not r.get("success", True)],
        }
    if human and n and all(r["output"] in human for r in rows):
        # human[output] = {"no_critical": bool, "publishable": 1..5, "rater": R}
        # R must be "owner" or "independent_human" to count as HUMAN_EVALUATED.
        # "agent_eye" (an agent's own visual review) is recorded but is NOT a
        # human rating: status becomes AGENT_EYE_REVIEWED and pass stays None.
        raters = {human[r["output"]].get("rater", "unspecified") for r in rows}
        hc = sum(1 for r in rows if r.get("success", True) and human[r["output"]]["no_critical"])
        hp = sum(1 for r in rows if r.get("success", True) and human[r["output"]]["no_critical"]
                 and human[r["output"]]["publishable"] >= 4)
        g = summary["gates"]
        summary["raters"] = sorted(raters)
        if raters <= {"owner", "independent_human"}:
            status, passes = "HUMAN_EVALUATED", (hc / n >= 0.80, hp / n >= 0.70)
        elif raters == {"agent_eye"}:
            status, passes = "AGENT_EYE_REVIEWED", (None, None)
        else:
            status, passes = "NOT_EVALUATED", (None, None)  # mixed/unknown raters
        if status != "NOT_EVALUATED":
            g["no_critical_alteration"].update(numerator=hc, rate=round(hc / n, 3),
                                               status=status, **{"pass": passes[0]})
            g["publish_ready"].update(numerator=hp, rate=round(hp / n, 3),
                                      status=status, **{"pass": passes[1]})
        summary["quality_status"] = status
    import statistics
    for route in {r["route"] for r in rows}:
        sub = [r for r in rows if r["route"] == route]
        scored = [r["score_fidelity"] for r in sub if r.get("success", True) and r.get("score_fidelity")]
        crit = sum(1 for r in sub if r["critical_errors"])
        # PDF economics: median latency, cost per successful output, retries.
        times = [r["seconds"] for r in sub if r.get("seconds")]
        successes = sum(1 for r in sub if r.get("success"))
        credits = sum(r.get("credits_estimated", 0) for r in sub)
        summary["routes"][route] = {
            "n": len(sub),
            "mean_fidelity": round(sum(scored) / len(scored), 2) if scored else None,
            "critical_rate": round(crit / len(sub), 3),
            "gate_80pct_no_critical": None,
            "automated_proxy_80pct_no_critical": (crit / len(sub)) <= 0.20,
            "quality_status": "NOT_EVALUATED",

            "median_seconds": round(statistics.median(times), 2) if times else None,
            "success_rate": round(successes / len(sub), 3),
            "success_rate_basis": "executor completion, not publication usability or product fidelity",
            "completed_outputs": successes,
            "failed_attempts": len(sub) - successes,
            "retries_needed": len(sub) - successes,
            "retries_needed_basis": "legacy field: failed attempts only; actual retry count is not recorded",
            "credits_total": credits,
            "credits_basis": "sum of credits_estimated; not a verified billing ledger",
            "credits_per_publishable_output": None,
            "publishable_cost_status": "NOT_EVALUATED: completed outputs are not verified publishable outputs",
            "credits_per_successful_output":
                round(credits / successes, 2) if credits and successes else None,
        }
    ranked = sorted(((v["mean_fidelity"], k) for k, v in summary["routes"].items()
                     if v["mean_fidelity"] is not None), reverse=True)
    if ranked:
        summary["config_comparison"] = {
            "best_quality_route": None,
            "fallback_route": None,
            "automated_fidelity_leader": ranked[0][1],
            "recommendation_status": "NOT_EVALUATED",
            "note": "Automated fidelity ordering only, not a quality or value "
                    "recommendation. A cheaper viable fallback needs human "
                    "quality and measured cost evidence."}
    return summary


def summarize_runs(run_dirs: list[Path], out: Path,
                   source_meta: dict[str, dict] | None = None) -> Path:
    """Aggregate EXISTING run manifests into the same gate report without
    re-running anything. Lets us re-score after QA changes (e.g. the matte
    tripwires) and reuse expensive GPU runs. source_meta maps manifest source
    path -> {label, license}; missing metadata is marked, not invented."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rd in run_dirs:
        manifest = rd / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        meta = (source_meta or {}).get(m.get("source"), {})
        for v in m.get("variants", []):
            q = v.get("qa", {})
            from .completion import saved_variant_failed
            failed = saved_variant_failed(v)
            row = {"source": m.get("source"),
                   "source_label": meta.get("label", "(unlabeled)"),
                   "source_sha256": m.get("source_provenance", {}).get("sha256"),
                   "source_classification": m.get("source_provenance", {}).get("classification", "UNKNOWN / unclassified legacy run"),
                   "source_match_basis": m.get("source_provenance", {}).get("match_basis", "unrecorded"),
                   "source_license": meta.get("license", "(unrecorded)"),
                   "route": v.get("route"), "template": q.get("template"),
                   "seed": v.get("seed"), "backend": v.get("backend", ""),
                   "seconds": v.get("seconds"),
                   "credits_estimated": v.get("credits_estimated", 0),
                   "success": not failed and bool(v.get("success", Path(v.get("output", "")).is_file())),
                   "score_fidelity": None if failed else q.get("score_fidelity"),
                   "critical_errors": q.get("critical_errors") or [],
                   "output": v.get("output")}
            rows.append(row)
    csv_path = out / "scores.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = _summarize(rows, len(run_dirs))
    (out / "benchmark_report.json").write_text(json.dumps(summary, indent=2))
    return csv_path
