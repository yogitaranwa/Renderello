"""Studio: the single high-level generate API shared by the website and the
Colab notebook. One call = the brief's working flow:

    product photo(s) -> preset -> optional long instruction + references
    -> N variants -> QA -> compare/download bundle

Two generation modes, chosen explicitly per run (no silent fallback):
- compose: deterministic cutout-compose. Scene providers create the
  environment only (procedural / sdxl_lightning / hf_inference); product
  pixels stay source-exact.
- magichour: whole-image edit through the Magic Hour editor API - the
  brief's intended backend, owner-directed 2026-11-20. Live calls need the
  key plus RENDERELLO_MH_LIVE=1, set only with a fresh owner spend
  approval; without configuration the run refuses with zero bill.
  RENDERELLO_MH_DEV_MOCK=1 is an explicit developer placeholder mode,
  labeled as such everywhere.

Every run writes a manifest recording provider, prompt plan, QA verdicts
and receipts. Nothing is hidden: failures are listed, not dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import compose, prompts, qa
from .longprompt import plan_prompt
from .multinput import (GroupMember, compose_group, cutout_products,
                        merged_reference_style)
from .presets import Preset, build_detailed_prompt, get_preset
from .scene_providers import SceneProvider, get_provider
from .schemas import Aspect


def runtime_versions() -> dict:
    """Installed package versions only; never environment variables or secrets."""
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors",
                 "huggingface_hub", "Pillow", "numpy", "onnxruntime", "gradio"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass
class StudioRequest:
    product_paths: list[Path]
    preset_key: str
    instruction: str = ""
    reference_paths: list[Path] = field(default_factory=list)
    variants: int = 4
    aspect: str | None = None          # "1:1" | "4:5" | "3:2" | None
    provider: str = "procedural"       # procedural | sdxl_lightning | hf_inference | magichour
    mh_route: str = "scene"            # scene = background-only + preserved cutout; whole = whole-image edit; hybrid = preserved cutout + harmonize pass
    material: str | None = None
    product_description: str = ""
    knob_values: dict[str, str] = field(default_factory=dict)
    device: str = "cpu"


@dataclass
class VariantOut:
    index: int
    seed: int
    image: str                    # path relative to run dir
    provider: str
    prompt_report: dict
    qa: dict
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class StudioResult:
    run_id: str
    run_dir: Path
    preset: str
    provider_requested: str
    variants: list[VariantOut]
    group_errors: list[str]
    reference_style: dict | None
    manifest_path: Path
    elapsed_s: float


_ASPECTS = {a.value: a for a in Aspect}


def _product_entries(product_paths, members: list[GroupMember]) -> list[dict]:
    """Manifest product rows: file identity plus which cutout backend actually
    produced the matte, so a silent fallback can never hide in a run."""
    by_name = {Path(m.path).name: m for m in members}
    entries: list[dict] = []
    for p in product_paths:
        name = Path(p).name
        entry = {"file": name, "sha16": _hash_file(Path(p))}
        m = by_name.get(name)
        if m is not None and m.cutout is not None:
            entry["cutout_backend"] = m.cutout.backend
            entry["cutout_coverage"] = round(m.cutout.coverage, 4)
            if getattr(m.cutout, "integrity", None) is not None:
                entry["matte_integrity"] = m.cutout.integrity
        entries.append(entry)
    return entries


def _cutout_warnings(members: list[GroupMember]) -> list[str]:
    """User-visible honesty lines when the learned cutout did not run or
    its matte measured fragmented (review-confirmed runs stay honest)."""
    warns: list[str] = []
    for m in members:
        if m.cutout is not None and getattr(m.cutout, "integrity", None):
            from .matte_integrity import assess as _ma
            _w = _ma(m.cutout.integrity)
            if _w:
                warns.append(f"{Path(m.path).name}: {_w}")
        if m.cutout is not None and m.cutout.backend == "grabcut":
            warns.append(
                f"{Path(m.path).name}: cutout fell back to OpenCV grabcut "
                "(learned BiRefNet failed - see the RuntimeWarning in the "
                "run output for the reason). Matte edges are crude, not "
                "publish-quality.")
    return warns


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


import threading
_BUDGET_LOCK = threading.Lock()


class _ledger_file_lock():
    """Cross-process exclusive lock on the ledger (fcntl flock). The thread
    lock above serializes this process; the file lock serializes concurrent
    workers on the same host (e.g. multiple uvicorn processes sharing the
    runs volume)."""
    def __enter__(self):
        import fcntl
        lock = _ledger_path().parent / (self._name() + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(lock, "a")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()

    @staticmethod
    def _name() -> str:
        return _ledger_path().name


def _ledger_path() -> Path:
    return Path(os.environ.get("RENDERELLO_MH_LEDGER",
                               "runs/mh-spend-ledger.jsonl"))


def _ledger_append(row: dict) -> None:
    ledger = _ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _ledger_spent() -> int:
    # Reservations count positive; reconciliations are negative adjustments,
    # so the sum is the current committed spend. A crashed batch keeps its
    # worst-case reservation - conservative, never under-counts.
    ledger = _ledger_path()
    spent = 0
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            amt = row.get("credits_spent")
            if isinstance(amt, (int, float)):
                spent += int(amt)
    return spent


def _budget_checks(model: str, need: int, require_floor: bool) -> None:
    """All refusal checks for a batch costing `need` credits. Raises
    RuntimeError with a zero-bill message on any failure."""
    from .backends.magichour import MODEL_TABLE
    raw = os.environ.get("RENDERELLO_MH_BUDGET_CREDITS", "").strip()
    if not raw:
        raise RuntimeError(
            "Magic Hour live generation is OFF: no credit budget allocated "
            "(RENDERELLO_MH_BUDGET_CREDITS unset). Zero credits billed.")
    try:
        budget = int(raw)
    except ValueError:
        raise RuntimeError("RENDERELLO_MH_BUDGET_CREDITS is not an integer; "
                           "live generation refused, zero credits billed.")
    listed = MODEL_TABLE.get(model, {}).get("credits")
    if listed is None:
        raise RuntimeError(f"unknown model {model!r}; live generation refused, "
                           "zero credits billed.")
    max_batch = int(os.environ.get("RENDERELLO_MH_MAX_BATCH_CREDITS", "120") or 120)
    if need > max_batch:
        raise RuntimeError(
            f"batch too large: this run could cost up to {need} credits "
            f"(limit {max_batch} per batch). Reduce variants/templates or "
            "the owner raises RENDERELLO_MH_MAX_BATCH_CREDITS. Refused - "
            "zero credits billed.")
    spent = _ledger_spent()
    if spent + need > budget:
        raise RuntimeError(
            f"Magic Hour budget exhausted: {spent}/{budget} credits committed, "
            f"this batch could cost up to {need}. Refused - zero credits "
            "billed. The owner must authorize more budget.")
    # Deploy-proof hard cap: the ledger lives on ephemeral storage and resets
    # on every redeploy, so the ledger check alone could re-allow spend
    # already consumed. RENDERELLO_MH_CAP_ANCHOR = "<anchor_balance>:<total>"
    # anchors the cap in the LIVE account balance, which cannot reset:
    #   spent_so_far = anchor_balance - current_balance
    # A balance ABOVE the anchor (top-up, credit adjustment) fails CLOSED:
    # spend freezes until the owner re-anchors - top-ups can never widen the
    # authorized total. Combined with the web layer's single-live-job rule,
    # every posted charge is reflected before the next reservation is read.
    # Durable ledger (GitLab CAS project): survives redeploys, unlike the
    # on-host JSONL. The effective spend is the MAX of every counter - no
    # counter is ever trusted to be complete on its own.
    from . import mh_ledger
    durable_spent = None
    if mh_ledger.enabled():
        durable_spent = mh_ledger.spent()  # raises (fail closed) on failure
        if durable_spent > spent:
            spent = durable_spent
            if spent + need > budget:
                raise RuntimeError(
                    f"Magic Hour budget exhausted: {spent}/{budget} credits "
                    f"committed (durable ledger), this batch could cost up "
                    f"to {need}. Refused - zero credits billed. The owner "
                    "must authorize more budget.")
    anchor_raw = os.environ.get("RENDERELLO_MH_CAP_ANCHOR", "").strip()
    floor_raw = os.environ.get("RENDERELLO_MH_MIN_BALANCE", "").strip()
    if not anchor_raw and not floor_raw:
        if require_floor:
            raise RuntimeError(
                "RENDERELLO_MH_CAP_ANCHOR unset: the balance-anchored hard "
                "cap is required for live spend (the ledger resets on "
                "redeploy). Refused - zero credits billed.")
        return
    try:
        from .backends.magichour import MagicHourBackend
        bal = MagicHourBackend(model=model, live=True).balance()
        balance = int(bal.get("credits", 0))
    except Exception as e:
        raise RuntimeError(f"could not verify live account balance "
                           f"({type(e).__name__}); live generation "
                           "refused, zero credits billed.")
    if anchor_raw:
        try:
            a_bal_s, a_total_s = anchor_raw.split(":")
            a_bal, a_total = int(a_bal_s), int(a_total_s)
        except ValueError:
            raise RuntimeError("RENDERELLO_MH_CAP_ANCHOR must be "
                               "'<anchor_balance>:<authorized_total>'; "
                               "refused, zero credits billed.")
        spent_live = a_bal - balance
        _LAST_BALANCE["value"] = balance  # prime the per-call drop check
        if spent_live < 0:
            raise RuntimeError(
                "live balance is ABOVE the authorized anchor (top-up or "
                "adjustment?): spend frozen until the owner re-anchors. "
                "Refused - zero credits billed.")
        # Both counters in the SAME units: the durable ledger is seeded
        # with RENDERELLO_MH_SPENT_BASELINE (spend before the anchor was
        # set), so durable_total == baseline + spent_since_anchor when
        # nothing is lost. durable < baseline + spent_live means the
        # durable ledger lost history (wrong project, reseeded file,
        # manual edit) - freeze rather than guess. (durable > baseline +
        # spent_live is normal: reservations land in the durable ledger
        # before the charge posts to the account balance.)
        baseline_raw = os.environ.get("RENDERELLO_MH_SPENT_BASELINE", "0").strip()
        try:
            baseline = int(baseline_raw) if baseline_raw else 0
        except ValueError:
            raise RuntimeError("RENDERELLO_MH_SPENT_BASELINE is not an "
                               "integer; live generation refused, zero "
                               "credits billed.")
        if durable_spent is not None and durable_spent < baseline + spent_live:
            raise RuntimeError(
                "durable ledger disagrees with the balance anchor "
                f"(durable {durable_spent} < baseline {baseline} + "
                f"anchor-derived {spent_live}): spend frozen until the "
                "owner reconciles the counters. Refused - zero credits "
                "billed.")
        if spent_live + need > a_total:
            raise RuntimeError(
                f"Magic Hour hard cap reached: {spent_live}/{a_total} "
                f"credits used since anchor, this batch needs up to {need}. "
                "Refused - zero credits billed. The owner must authorize "
                "more budget.")
    elif floor_raw:
        try:
            floor = int(floor_raw)
        except ValueError:
            raise RuntimeError("RENDERELLO_MH_MIN_BALANCE is not an integer; "
                               "live generation refused, zero credits billed.")
        if balance - need < floor:
            raise RuntimeError(
                f"live balance floor reached: balance {balance}, batch needs "
                f"up to {need}, floor {floor}. Refused - zero credits billed. "
                "The owner must authorize more budget.")


_LAST_BALANCE = {"value": None}


def _live_model_allowlist() -> set:
    raw = os.environ.get("RENDERELLO_MH_LIVE_MODELS",
                         "seedream-v5-flash").strip()
    return {m.strip() for m in raw.split(",") if m.strip()}


def _post_call_charge(backend, listed: int, model: str) -> tuple:
    """Authoritative charge for ONE paid call: the LARGER of the provider's
    reported credits_charged (submission accept or terminal detail) and the
    actual account balance drop since the previous check. Returns
    (credits, abort_reason). abort_reason is set when the batch must stop:
    the balance dropped more than the provider reported, the observed
    charge exceeds the listed rate, or the balance is unreadable after a
    paid call. A smaller drop (refund/adjustment) NEVER reduces the
    counted spend."""
    obs = _observed_charge(backend)
    prev = _LAST_BALANCE["value"]
    try:
        from .backends.magichour import MagicHourBackend
        bal = MagicHourBackend(model=model, live=True).balance()
        now = int(bal.get("credits", 0))
    except Exception as e:
        return (obs if obs is not None else listed), (
            f"post-call balance unreadable ({type(e).__name__}) after a "
            "paid call: batch aborted; the counted charge above IS "
            "reconciled to both ledgers (not dropped); further spend "
            "fails closed until the balance reads again")
    _LAST_BALANCE["value"] = now
    if prev is not None:
        drop = prev - now
        if drop > (obs if obs is not None else 0):
            return drop, (
                f"account balance dropped {drop} credits but the provider "
                f"reported {obs}: the larger figure is counted and the "
                "batch aborted")
    if obs is not None and obs > listed:
        return obs, (
            f"provider charged {obs} credits for this call, above the "
            f"listed {listed}: batch aborted, remaining variants not "
            "attempted")
    return (obs if obs is not None else listed), None


def reserve_budget(model: str, count: int) -> int:
    """Atomic check + reservation of the worst-case batch cost. The lock
    closes the check-then-spend race between concurrent jobs; the durable
    ledger row stands even if the process dies mid-batch (conservative).
    Returns the reserved credit amount."""
    from .backends.magichour import MODEL_TABLE
    if model not in _live_model_allowlist():
        raise RuntimeError(
            f"model {model!r} is not on the live allowlist "
            "(RENDERELLO_MH_LIVE_MODELS, default seedream-v5-flash: the "
            "only model whose exact request cost is source-verified at "
            "the pinned 640px / single-image shape). Refused - zero "
            "credits billed.")
    listed = MODEL_TABLE.get(model, {}).get("credits")
    if listed is None:
        raise RuntimeError(f"unknown model {model!r}; live generation refused, "
                           "zero credits billed.")
    need = listed * max(1, count)
    with _BUDGET_LOCK, _ledger_file_lock():
        _budget_checks(model, need, require_floor=True)
        # Durable ledger FIRST: the reservation must survive a redeploy
        # before any paid call can happen. A failed durable reserve fails
        # closed - the paid call never happens (LedgerError propagates).
        from . import mh_ledger
        if mh_ledger.enabled():
            mh_ledger.reserve(need)
        _ledger_append({"model": model, "credits_spent": need,
                        "kind": "reserve", "charged_observed": False})
    return need


def reconcile_budget(model: str, reserved: int, actual: int) -> None:
    """After a batch: adjust the reservation to the observed (or
    conservatively listed) actual spend."""
    # Unknown observed charge: keep the full reservation (conservative).
    if actual is None:
        return
    delta = int(actual) - int(reserved)
    if not delta:
        return
    with _BUDGET_LOCK, _ledger_file_lock():
        from . import mh_ledger
        if mh_ledger.enabled():
            try:
                # Move the reservation TO the observed charge - DOWN when
                # lower, UP when the provider overcharged: an overcharge
                # is real spend the durable ledger must record, or a
                # restart would permanently under-count it.
                mh_ledger.reconcile(reserved, actual)
            except mh_ledger.LedgerError:
                # Failed reconcile KEEPS the reservation in both ledgers.
                # If the real charge was higher, the balance anchor
                # exceeds the durable ledger and spend freezes until the
                # owner reconciles the counters - fail closed either way.
                return
        _ledger_append({"model": model, "credits_spent": delta,
                        "kind": "reconcile",
                        "charged_observed": True,
                        "overcharge": bool(delta > 0)})


def _enforce_budget(model: str, count: int = 1) -> None:
    """Check-only pre-flight (no reservation): used by the web layer to fail
    fast before upload work. The authoritative control is reserve_budget."""
    from .backends.magichour import MODEL_TABLE
    listed = MODEL_TABLE.get(model, {}).get("credits")
    if listed is None:
        raise RuntimeError(f"unknown model {model!r}; live generation refused, "
                           "zero credits billed.")
    _budget_checks(model, listed * max(1, count), require_floor=False)


def _observed_charge(backend) -> int | None:
    """Highest charge the provider reported for the last call, from the
    submission accept (authoritative at POST time per the API reference:
    the 200 response carries credits_charged) or the terminal detail.
    None when neither was observed (unsettled)."""
    state = backend.last_result or {}
    obs = None
    for row in (state.get("submission"), state.get("final")):
        if isinstance(row, dict):
            c = row.get("credits_charged")
            if isinstance(c, (int, float)) and c >= 0:
                obs = max(obs or 0, int(c))
    return obs


class _OverchargeError(RuntimeError):
    """The provider charged above the listed reservation for one call.
    Raised to abort the rest of a batch immediately."""


SCENE_ATTEMPTS_DEFAULT = 1
SCENE_ATTEMPTS_MAX = 3


def scene_attempts_from(knobs: dict) -> int:
    """Single parser for the scene_max_attempts knob, shared by the route and
    the web layer so reservations can never undercount: clamp to 1..3, any
    junk/negative/oversize value falls back to the default."""
    try:
        n = int(str(knobs.get("scene_max_attempts", SCENE_ATTEMPTS_DEFAULT)))
    except (TypeError, ValueError):
        return SCENE_ATTEMPTS_DEFAULT
    if n < 1 or n > SCENE_ATTEMPTS_MAX:
        return SCENE_ATTEMPTS_DEFAULT
    return n


def budget_status(model: str = "seedream-v5-flash") -> dict:
    """Current budget view for UIs: allocated, spent, remaining, per-image
    listed cost. Never raises; unconfigured budget reports zeros."""
    from .backends.magichour import MODEL_TABLE
    raw = os.environ.get("RENDERELLO_MH_BUDGET_CREDITS", "").strip()
    try:
        budget = int(raw) if raw else 0
    except ValueError:
        budget = 0
    spent = _ledger_spent()
    durable = None
    try:
        from . import mh_ledger
        if mh_ledger.enabled():
            durable = mh_ledger.spent()
            spent = max(spent, durable)
    except Exception:
        pass  # status is informational; the reserve path fails closed
    out = {"allocated": budget, "spent": spent,
           "remaining": max(0, budget - spent),
           "credits_per_image": MODEL_TABLE.get(model, {}).get("credits"),
           "max_batch": int(os.environ.get("RENDERELLO_MH_MAX_BATCH_CREDITS",
                                           "120") or 120)}
    if durable is not None:
        out["durable_spent"] = durable
    return out


def record_spend(model: str, charged, listed: int) -> None:
    """Append one line to the aggregate spend ledger after a live call.
    Unknown charges are recorded at the listed rate (conservative)."""
    ledger = Path(os.environ.get("RENDERELLO_MH_LEDGER",
                                 "runs/mh-spend-ledger.jsonl"))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    amt = charged if isinstance(charged, (int, float)) and charged >= 0 else listed
    with ledger.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "model": model, "credits_spent": int(amt),
                            "charged_observed": isinstance(charged, (int, float))}) + "\n")


def _qa_dict(report) -> dict:
    return {
        "fidelity_score": report.score_fidelity,
        "critical_errors": list(report.critical_errors),
        "notes": report.notes,
        "metrics": {k: (round(float(x), 4) if x is not None else None)
                    for k, x in (("ssim_product", report.ssim_product),
                                 ("l1_product", report.l1_product),
                                 ("color_drift", report.color_drift),
                                 ("edge_retained", report.edge_retained))},
    }


def _save_contact_sheet(run_dir: Path, variants: list[VariantOut],
                        members: list[GroupMember]) -> None:
    """Side-by-side source(s) + variants for quick eye review/download."""
    tiles: list[Image.Image] = []
    for m in members:
        if m.cutout is not None:
            tiles.append(Image.open(m.path).convert("RGB"))
    for v in variants:
        if v.image and (run_dir / v.image).exists():
            tiles.append(Image.open(run_dir / v.image).convert("RGB"))
    if not tiles:
        return
    h = 420
    scaled = []
    for t in tiles:
        w = max(1, int(t.width * h / t.height))
        scaled.append(t.resize((w, h), Image.LANCZOS))
    sheet = Image.new("RGB", (sum(t.width for t in scaled) + 8 * len(scaled),
                              h + 8), (24, 24, 24))
    x = 8
    for t in scaled:
        sheet.paste(t, (x, 4))
        x += t.width + 8
    sheet.save(run_dir / "contact_sheet.png")


def run_studio(req: StudioRequest, out_root: Path,
               provider: SceneProvider | None = None) -> StudioResult:
    """Run the full flow once. CPU-safe; GPU/live providers raise honestly."""
    t0 = time.time()
    preset: Preset = get_preset(req.preset_key)
    template = preset.template
    prompts.validate_instruction(req.instruction)
    aspect = _ASPECTS[req.aspect] if req.aspect else template.aspect

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(
        (str(req.product_paths) + str(time.time())).encode()).hexdigest()[:6]
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Reference style (never composited; steers scene/prompt only).
    ref_style = merged_reference_style([Path(p) for p in req.reference_paths])
    ref_fragment = ref_style.prompt_fragment() if ref_style else ""

    # Product cutouts (one per uploaded product photo). Editor mode does not
    # need them, but they are cheap and power the QA matte + honest errors.
    inter = run_dir / "intermediates"
    members = cutout_products([Path(p) for p in req.product_paths], inter,
                              device=req.device)
    group_errors = [f"{Path(m.path).name}: {m.error}" for m in members if m.error]
    ok_members = [m for m in members if m.cutout is not None]

    if req.provider == "magichour" and req.mh_route in ("scene", "hybrid"):
        if len(req.product_paths) != 1:
            raise RuntimeError("scene-compose requires exactly one product photo; no credits reserved or spent")
        if not ok_members:
            raise RuntimeError("no product cutout succeeded: " + "; ".join(group_errors))

    # Matte integrity (21 Nov 2026): measure component structure of every
    # cutout and record it on the manifest product row. Paid Magic Hour
    # routes fail closed (zero bill, before any reservation) when a matte
    # looks fragmented and the reviewer has not confirmed it; procedural
    # routes only warn. Thresholds and the measured table they come from
    # are documented in matte_integrity.py.
    from .matte_integrity import matte_stats, assess as matte_assess, assess_compact
    matte_review: list[str] = []
    for m in ok_members:
        _alpha = np.asarray(Image.open(m.cutout.matte_path).convert("L"))
        m.cutout.integrity = matte_stats(_alpha)
        _warn = matte_assess(m.cutout.integrity)
        if _warn:
            matte_review.append(f"{Path(m.path).name}: {_warn}")
    if matte_review and req.provider == "magichour" and req.mh_route in ("scene", "hybrid") \
            and req.knob_values.get("matte_reviewed") != "1":
        _short = "; ".join(
            f"{Path(m.path).name}: {assess_compact(m.cutout.integrity)}"
            for m in ok_members if assess_compact(m.cutout.integrity))
        # Prefix must stay in webapp _SAFE_ERROR_PREFIXES and the whole
        # message under its 300-char passthrough cap.
        from .matte_integrity import MatteReviewRequired
        previews = []
        for _i, _m in enumerate(ok_members):
            _src = Image.open(_m.path).convert("RGB")
            _a = Image.open(_m.cutout.matte_path).convert("L")
            _src.thumbnail((520, 520))
            _a = _a.resize(_src.size)
            _panel = Image.new("RGB", (_src.width * 2, _src.height), "white")
            _panel.paste(_src, (0, 0))
            _panel.paste(_a.convert("RGB"), (_src.width, 0))
            _name = f"matte_review_{_i}.png"
            _panel.save(run_dir / _name)
            previews.append(_name)
        raise MatteReviewRequired(
            "matte review required (no credits reserved or spent; review the "
            "saved matte, then re-run with knob matte_reviewed=1 to proceed): "
            + _short, run_dir.name, previews)

    if req.provider == "magichour":
        if req.mh_route not in ("scene", "whole", "hybrid"):
            raise ValueError(f"unknown Magic Hour route {req.mh_route!r}; "
                             "expected scene, whole or hybrid")
        if req.mh_route == "whole":
            return _run_magichour(req, preset, aspect, run_id, run_dir,
                                  ref_style, ref_fragment, members, group_errors, t0)
        return _run_magichour_scene(req, preset, aspect, run_id, run_dir,
                                    ref_style, ref_fragment, members,
                                    group_errors, t0,
                                    harmonize=(req.mh_route == "hybrid"))

    provider = provider or get_provider(req.provider)
    p_ok, p_why = provider.available()

    variants: list[VariantOut] = []
    n = max(1, min(req.variants, 4))
    for i in range(n):
        seed = int(req.knob_values.get("seed", 0) or 0) + i * 101
        full_prompt = build_detailed_prompt(
            preset, product_description=req.product_description,
            material=req.material, instruction=req.instruction,
            reference_style=ref_fragment, knob_values=req.knob_values)
        plan = plan_prompt(full_prompt, prompts.build_negative_prompt(),
                           provider.name)
        v = VariantOut(index=i, seed=seed, image="", provider=provider.name,
                       prompt_report=plan.report(), qa={})
        out_path = run_dir / f"variant_{i}_{provider.name}.png"
        try:
            if not ok_members:
                raise RuntimeError("no product cutout succeeded: " +
                                   "; ".join(group_errors))
            if not p_ok:
                raise RuntimeError(f"provider unavailable: {p_why}")
            if len(ok_members) > 1:
                canvas = compose_group(members, template, aspect, seed,
                                       reference_style=ref_style)
                v.warnings.append("group layout on procedural backdrop")
            else:
                co = ok_members[0].cutout
                scene = provider.generate_scene(
                    aspect.size, plan.full_text, seed, template,
                    negative=plan.negative)
                # Empty scene replaces the procedural backdrop; the verified
                # product cutout is pasted source-exact on top, with the
                # template's contact-shadow language (composite reference path).
                canvas_img = compose.composite(
                    co, template, aspect, seed,
                    reflection=(template.key == "reflective_surface"),
                    reference=np.asarray(scene))
                canvas = np.asarray(canvas_img)
            Image.fromarray(canvas).save(out_path)
            v.image = out_path.name
            src_img = Image.open(ok_members[0].path).convert("RGB")
            report = qa.evaluate(src_img, out_path, route="cutout_compose",
                                 template=preset.key, seed=seed)
            v.qa = _qa_dict(report)
            if len(ok_members) == 1:
                # Read the saved output, not a QA-resized comparison baseline.
                native = compose.audit_composite(
                    co, np.asarray(src_img),
                    np.asarray(Image.open(out_path).convert("RGB")),
                    aspect, seed, template.key)
                v.qa["native_pixel_audit"] = native
                if native["status"] != "MATCH":
                    v.warnings.append(
                        "native source RGB exactness not established: " + native["status"] +
                        "; edges and lighting still require eye review")
        except Exception as e:
            v.error = f"{type(e).__name__}: {e}"
        variants.append(v)

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset": preset.key,
        "provider_requested": req.provider,
        "provider_used": provider.name,
        "provider_available": p_ok,
        "provider_note": provider.quality_note,
        "mode": "cutout-compose",
        "products": _product_entries(req.product_paths, members),
        "warnings": _cutout_warnings(members),
        "references": [Path(p).name for p in req.reference_paths],
        "reference_style": (asdict(ref_style) if ref_style else None),
        "instruction": req.instruction,
        "material": req.material,
        "knobs": req.knob_values,
        "aspect": aspect.value,
        "group_errors": group_errors,
        "variants": [asdict(v) for v in variants],
        "elapsed_s": round(elapsed, 2),
        "contact_sheet": "contact_sheet.png",
        "honesty": ("Cutout routes paste source pixels but may resize them; native RGB fidelity requires the per-variant audit. "
                    "Background generation does not prove edge or lighting quality. QA scores "
                    "are automated proxies - human eye review remains the "
                    "acceptance gate per the brief."),
    }
    manifest_path = run_dir / "manifest.json"
    manifest["runtime_versions"] = runtime_versions()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _save_contact_sheet(run_dir, variants, members)

    return StudioResult(
        run_id=run_id, run_dir=run_dir, preset=preset.key,
        provider_requested=req.provider, variants=variants,
        group_errors=group_errors,
        reference_style=(asdict(ref_style) if ref_style else None),
        manifest_path=manifest_path, elapsed_s=elapsed)


def _run_magichour(req: StudioRequest, preset: Preset, aspect: Aspect,
                   run_id: str, run_dir: Path, ref_style, ref_fragment: str,
                   members: list[GroupMember], group_errors: list[str],
                   t0: float) -> StudioResult:
    """Whole-image edit route through the Magic Hour editor API.

    Live calls require the key in the environment AND RENDERELLO_MH_LIVE=1,
    which is set only with a fresh owner spend approval. Without that, the
    run REFUSES with a configuration-required message and zero bill - fake
    outputs never appear in the user flow. A synthetic placeholder exists
    only for explicit developer testing with RENDERELLO_MH_DEV_MOCK=1, and
    is labeled as such in every manifest/variant. Products AND references
    are passed as editor input images (the API supports up to 9 on
    nano-banana/seedream families), so multi-product shots and style
    references work natively.
    """
    from .backends.magichour import MagicHourBackend
    model = req.knob_values.get("mh_model", "nano-banana-2.1")
    live = os.environ.get("RENDERELLO_MH_LIVE") == "1"
    dev_mock = os.environ.get("RENDERELLO_MH_DEV_MOCK") == "1"
    if not live and not dev_mock:
        raise RuntimeError(
            "Magic Hour is not configured for live generation on this "
            "server (no API key / live flag). Configuration required - "
            "zero credits were billed and no image was generated. "
            "Developers: RENDERELLO_MH_DEV_MOCK=1 runs the explicit "
            "placeholder test mode.")
    reserved = 0
    actual = {"n": 0}
    n = max(1, min(req.variants, 4))
    if live:
        # Only a SINGLE-INPUT request has source-verified exact pricing
        # (30 credits, seedream-v5-flash @ 640px, image_count=1). Extra
        # product images and reference images make the request multi-input
        # with unverified cost - refuse BEFORE any paid call.
        if len(req.product_paths) > 1 or req.reference_paths:
            raise RuntimeError(
                "live Magic Hour accepts exactly ONE input image: extra "
                "product images and reference images are refused before "
                "any paid call until multi-input pricing is verified. "
                "Zero credits billed.")
        reserved = reserve_budget(model, n)
    backend = MagicHourBackend(model=model, live=live,
                               receipt_path=run_dir / "mh-receipts.jsonl")
    variants: list[VariantOut] = []
    try:
      for i in range(n):
        seed = int(req.knob_values.get("seed", 0) or 0) + i * 101
        full_prompt = build_detailed_prompt(
            preset, product_description=req.product_description,
            material=req.material, instruction=req.instruction,
            reference_style=ref_fragment, knob_values=req.knob_values)
        plan = plan_prompt(full_prompt, prompts.build_negative_prompt(),
                           "magichour")
        v = VariantOut(index=i, seed=seed, image="", provider="magichour",
                       prompt_report=plan.report(), qa={})
        out_path = run_dir / f"variant_{i}_magichour.png"
        spend_recorded = False
        try:
            if not req.product_paths:
                raise RuntimeError("no product images supplied")
            src = Image.open(req.product_paths[0]).convert("RGB")
            extras = [Image.open(p).convert("RGB")
                      for p in list(req.product_paths[1:]) + list(req.reference_paths)]
            edited = backend.edit(src, plan.full_text, plan.negative, seed,
                                  aspect, extra_images=extras)
            if live:
                from .backends.magichour import MODEL_TABLE
                listed = MODEL_TABLE[model]["credits"]
                charge, abort_reason = _post_call_charge(backend, listed,
                                                         model)
                actual["n"] += charge
                spend_recorded = True
                if abort_reason:
                    v.warnings.append(abort_reason)
                    edited.save(out_path)
                    v.image = out_path.name
                    variants.append(v)
                    break
            edited.save(out_path)
            v.image = out_path.name
            v.warnings.append(
                "whole-image AI edit: product pixels are regenerated, not "
                "source-exact - verify label/logo fidelity before publishing")
            report = qa.evaluate(src, out_path, route="pure_prompt",
                                 template=preset.key, seed=seed)
            v.qa = _qa_dict(report)
            if backend.last_result and backend.last_result.get("submission"):
                sub = backend.last_result["submission"]
                v.warnings.append(
                    f"project {sub.get('id')} "
                    f"credits_charged={sub.get('credits_charged')}")
        except Exception as e:
            if live and not spend_recorded:
                # A failed live call may still have been billed (transport
                # error after POST, failed poll/download). Use the
                # submission-accept observed charge when the provider
                # reported one; the listed rate only as the last fallback.
                from .backends.magichour import MODEL_TABLE
                obs = _observed_charge(backend)
                actual["n"] += (obs if obs is not None
                                else MODEL_TABLE[model]["credits"])
            v.error = f"{type(e).__name__}: {e}"
        variants.append(v)
    finally:
        if live:
            # An exception escaping the loop (anything not handled per
            # variant, incl. BaseException) may have interrupted a call that
            # was already accepted: keep the full reservation (unknown
            # charge) instead of refunding it.
            reconcile_budget(model, reserved,
                             actual["n"] if sys.exc_info()[0] is None else None)
    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset": preset.key, "provider_requested": req.provider,
        "provider_used": "magichour", "mode": "whole-image edit",
        "model": model, "live": backend.live,
        "credits_note": ("DEVELOPER PLACEHOLDER MODE - synthetic gradient, not a real generation, zero credits" if not backend.live else
                         "see mh-receipts.jsonl for charged amounts"),
        "products": _product_entries(req.product_paths, members),
        "warnings": _cutout_warnings(members),
        "references": [Path(p).name for p in req.reference_paths],
        "reference_style": (asdict(ref_style) if ref_style else None),
        "instruction": req.instruction, "material": req.material,
        "knobs": req.knob_values, "aspect": aspect.value,
        "group_errors": group_errors,
        "variants": [asdict(v) for v in variants],
        "elapsed_s": round(elapsed, 2),
        "contact_sheet": "contact_sheet.png",
        "honesty": ("Magic Hour edits regenerate the whole image including "
                    "the product; QA scores are automated proxies and human "
                    "eye review is the acceptance gate per the brief."),
    }
    manifest_path = run_dir / "manifest.json"
    manifest["runtime_versions"] = runtime_versions()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _save_contact_sheet(run_dir, variants, members)
    return StudioResult(
        run_id=run_id, run_dir=run_dir, preset=preset.key,
        provider_requested=req.provider, variants=variants,
        group_errors=group_errors,
        reference_style=(asdict(ref_style) if ref_style else None),
        manifest_path=manifest_path, elapsed_s=elapsed)


def _run_magichour_scene(req: StudioRequest, preset: Preset, aspect: Aspect,
                         run_id: str, run_dir: Path, ref_style, ref_fragment: str,
                         members: list[GroupMember], group_errors: list[str],
                         t0: float, harmonize: bool = False) -> StudioResult:
    """Scene-compose Magic Hour route: the paid model generates ONLY the
    background scene; the verified product cutout is pasted back source-exact
    with a grounded contact shadow. Product pixels are never the edit target.

    harmonize=True restores the 13 Nov 2026 hybrid route: after the
    source-exact paste, ONE extra paid edit pass blends lighting, shadow and
    grading so the product sits naturally (product-fidelity prompt). The
    harmonize pass CAN repaint product pixels, so its output is flagged for
    mandatory eye review and region-drift QA.

    Live calls require the key, RENDERELLO_MH_LIVE=1, and the whole batch
    (variants x scene attempts, plus one harmonize call per variant when
    enabled) must fit the aggregate budget - otherwise the run refuses with
    zero bill. Generated scenes with long straight-edge intrusions (softbox
    frames) or a suspected object in the placement area are rejected and
    regenerated, capped by SCENE_MAX_ATTEMPTS; each attempt is a paid
    generation and is recorded in the ledger whether it succeeds or fails.
    """
    from .backends.magichour import MagicHourBackend, MODEL_TABLE
    from .scene_acceptance import long_straight_edges
    template = preset.template
    model = req.knob_values.get("mh_model", "seedream-v5-flash")
    live = os.environ.get("RENDERELLO_MH_LIVE") == "1"
    dev_mock = os.environ.get("RENDERELLO_MH_DEV_MOCK") == "1"
    if not live and not dev_mock:
        raise RuntimeError(
            "Magic Hour is not configured for live generation on this "
            "server (no API key / live flag). Configuration required - "
            "zero credits were billed and no image was generated. "
            "Developers: RENDERELLO_MH_DEV_MOCK=1 runs the explicit "
            "placeholder test mode.")
    max_attempts = scene_attempts_from(req.knob_values)
    n = max(1, min(req.variants, 4))
    from .empty_scene_prompt import build_empty_scene_prompt, UnsupportedEmptySceneStaging
    try:
        scene_prompt = build_empty_scene_prompt(
            template, aspect_key=aspect.value,
            instruction=req.instruction)
    except UnsupportedEmptySceneStaging:
        raise
    except ValueError:
        # The instruction named the product: it cannot reach the scene
        # model without re-creating the contradictory prompt. Drop it
        # from the scene prompt and say so on the variant.
        scene_prompt = build_empty_scene_prompt(
            template, aspect_key=aspect.value)
        _scene_note_dropped = True
    else:
        _scene_note_dropped = False
    reserved = 0
    actual = {"n": 0}
    if live:
        # Worst case: every variant needs every scene attempt, plus one
        # harmonize edit per variant on the hybrid route.
        reserved = reserve_budget(model, n * (max_attempts + int(harmonize)))
    backend = MagicHourBackend(model=model, live=live,
                               receipt_path=run_dir / "mh-receipts.jsonl")
    ok_members = [m for m in members if m.cutout is not None]
    variants: list[VariantOut] = []
    try:
      for i in range(n):
        seed = int(req.knob_values.get("seed", 0) or 0) + i * 101
        full_prompt = build_detailed_prompt(
            preset, product_description=req.product_description,
            material=req.material, instruction=req.instruction,
            reference_style=ref_fragment, knob_values=req.knob_values)
        plan = plan_prompt(full_prompt, prompts.build_negative_prompt(),
                           "magichour")
        # The model receives ONLY the environment: a long, detailed,
        # non-contradictory empty-set prompt. Product detail stays in
        # plan.full_text for the report; it never reaches the scene model
        # (the 20 Nov 2026 duplicate-product failure came from sending the
        # product-bearing prompt with a contradictory 'no product' tail).
        v = VariantOut(index=i, seed=seed, image="", provider="magichour",
                       prompt_report=plan.report(), qa={})
        out_path = run_dir / f"variant_{i}_magichour_scene.png"
        attempt = {"n": 0}

        def scene_fn(canvas, _prompt, _seed=seed, _attempt=attempt):
            _attempt["n"] += 1
            if dev_mock and not live:
                return np.asarray(canvas)  # placeholder: gray canvas unchanged
            try:
                out = backend.edit(Image.fromarray(canvas),
                                   prompt=scene_prompt,
                                   negative_prompt="", seed=_seed + _attempt["n"],
                                   aspect=aspect)
            except Exception:
                # Failed after a possible accept: count the submission-accept
                # observed charge when reported, listed only as fallback.
                obs = _observed_charge(backend)
                actual["n"] += (obs if obs is not None
                                else MODEL_TABLE[model]["credits"])
                raise
            charge, abort_reason = _post_call_charge(
                backend, MODEL_TABLE[model]["credits"], model)
            actual["n"] += charge
            if abort_reason:
                # Overcharge / balance mismatch / unreadable balance: the
                # charge is counted above; stop this scene and the batch.
                raise _OverchargeError(abort_reason)
            return np.asarray(out.convert("RGB"))

        try:
            if not ok_members:
                raise RuntimeError("no product cutout succeeded: " +
                                   "; ".join(group_errors))
            if len(ok_members) > 1:
                raise RuntimeError(
                    "scene-compose composites one product per run; "
                    "multi-product groups are not supported on the Magic "
                    "Hour route - run products separately")
            co = ok_members[0].cutout
            img, diag = compose.composite_with_scene(
                co, preset.template, aspect, seed, scene_fn,
                prompt=scene_prompt,
                shadow_style=preset.template.shadow,
                scene_auto_lift=True,
                cache_dir=run_dir / "intermediates" / f"scene_layers_seed{seed}",
                scene_accept=long_straight_edges,
                scene_max_attempts=max_attempts,
                conditioning="empty_guide")
            if _scene_note_dropped:
                v.warnings.append(
                    "custom instruction named the product or in-frame equipment: excluded from the "
                    "empty-scene prompt to keep it non-contradictory; "
                    "environment-only wording was used")
            if isinstance(img, np.ndarray):
                Image.fromarray(img).save(out_path)
            else:
                img.save(out_path)
            # Per-run native pixel audit: the placed opaque product pixels
            # must equal the source RGB at the cutout bbox - measured, not
            # asserted. Size change or mismatch lands in the manifest with
            # its exact status; MATCH alone is not a quality acceptance.
            from .native_pixel_audit import audit_native_pixels
            _rgba = np.array(Image.open(co.rgba_path).convert("RGBA"))
            _bx, _by, _bw, _bh = co.bbox
            _px, _py, _pw, _ph = diag["placement"]
            native_audit = audit_native_pixels(
                np.array(Image.open(ok_members[0].path).convert("RGB")),
                _rgba[..., 3], np.array(Image.open(out_path).convert("RGB")),
                (int(_bx), int(_by), int(_bw), int(_bh)),
                (int(_px), int(_py), int(_pw), int(_ph)))
            if harmonize:
                # Hybrid route: ONE extra paid pass over the correct
                # composite to blend lighting/shadow/grading. Fidelity guard
                # first; product repaint risk is disclosed and eye review is
                # mandatory - this output is NOT source-exact by design.
                harm_prompt = (
                    prompts.FIDELITY_GUARD +
                    " This is already the correct product cutout on a new "
                    "background. Only harmonize lighting, shadow and color "
                    "grading so the product sits naturally in the scene "
                    f"({template.lighting}; {template.scene}). Do not move, "
                    "resize, rotate or repaint the product.")
                pre_harm_path = out_path
                out_path = run_dir / f"variant_{i}_magichour_hybrid.png"
                with Image.open(pre_harm_path) as _pre:
                    _pre.convert("RGB").save(out_path)
                try:
                    edited = backend.edit(Image.open(out_path).convert("RGB"),
                                          prompt=harm_prompt,
                                          negative_prompt="", seed=seed + 500,
                                          aspect=aspect)
                except Exception:
                    if live:
                        obs = _observed_charge(backend)
                        actual["n"] += (obs if obs is not None
                                        else MODEL_TABLE[model]["credits"])
                    raise
                if live:
                    charge, abort_reason = _post_call_charge(
                        backend, MODEL_TABLE[model]["credits"], model)
                    actual["n"] += charge
                    if abort_reason:
                        edited.convert("RGB").save(out_path)
                        v.image = out_path.name
                        v.warnings.append(abort_reason)
                        variants.append(v)
                        break
                edited.convert("RGB").save(out_path)
                # Post-harmonize audit: the harmonize pass may repaint the
                # product, so the FINAL image gets its own comparison,
                # recorded as drift - never inherited from the pre-pass MATCH.
                post_harm_audit = audit_native_pixels(
                    np.array(Image.open(ok_members[0].path).convert("RGB")),
                    _rgba[..., 3],
                    np.array(Image.open(out_path).convert("RGB")),
                    (int(_bx), int(_by), int(_bw), int(_bh)),
                    (int(_px), int(_py), int(_pw), int(_ph)))
                post_harm_audit["interpretation"] = (
                    "POST_HARMONIZE_DRIFT: mismatch expected - the model "
                    "blends lighting/grading over the composite; MATCH on "
                    "the final image is not claimed and not required. Eye "
                    "review against the pre-harmonize composite is the gate.")
                v.warnings.append(
                    "hybrid harmonize pass: lighting/shadow/grading blended "
                    "by the model over the source-exact composite; product "
                    "pixels may have been repainted - EYE REVIEW REQUIRED, "
                    "compare against the pre-harmonize composite "
                    f"{pre_harm_path.name}")
            v.image = out_path.name
            v.warnings.append(
                "product-preserving route: only the background is generated; "
                "product pixels come from the source cutout (scaled to the "
                "template placement, not resampled by the model)")
            if diag.get("scene_acceptance", {}).get("intrusion"):
                v.warnings.append(
                    "scene screen still flags an intrusion after the configured call limit; "
                    "no further paid scene call was made; DO NOT treat this as a quality pass")
            if attempt["n"] > 1:
                v.warnings.append(
                    f"scene regenerated {attempt['n']}x: straight-edge "
                    "artifact rejection fired (softbox/frame intrusion)")
            v.warnings.append(
                "eye review required - QA heuristics are screens, not proof")
            src_img = Image.open(ok_members[0].path).convert("RGB")
            report = qa.evaluate(src_img, out_path, route="cutout_compose",
                                 template=preset.key, seed=seed)
            v.qa = _qa_dict(report)
            v.qa["native_pixel_audit"] = native_audit
            if harmonize:
                v.qa["native_pixel_audit_purpose"] = "PRE_HARMONIZE_COMPOSITE"
                v.qa["post_harmonize_audit"] = post_harm_audit
            v.qa["scene_diag"] = {k: diag[k] for k in (
                "ring_vs_far_luma_diff", "needs_eye_review",
                "publication_status", "scene_acceptance", "conditioning",
                "emptiness_screen") if k in diag}
        except Exception as e:
            if live and attempt["n"] == 0 and not isinstance(e, _OverchargeError):
                # The call may have been billed despite the error.
                actual["n"] += MODEL_TABLE[model]["credits"]
            v.error = f"{type(e).__name__}: {e}"
            if isinstance(e, _OverchargeError):
                # Overcharge already counted at the call; stop the batch.
                variants.append(v)
                break
        variants.append(v)
    finally:
        if live:
            # An exception escaping the loop (anything not handled per
            # variant, incl. BaseException) may have interrupted a call that
            # was already accepted: keep the full reservation (unknown
            # charge) instead of refunding it.
            reconcile_budget(model, reserved,
                             actual["n"] if sys.exc_info()[0] is None else None)
    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset": preset.key, "provider_requested": req.provider,
        "provider_used": "magichour",
        "mode": ("hybrid (background-only generation + preserved cutout + "
                 "harmonize pass; product repaint possible - eye review)"
                 if harmonize else
                 "scene-compose (background-only generation + preserved cutout)"),
        "model": model, "live": backend.live,
        "credits_note": ("DEVELOPER PLACEHOLDER MODE - no live calls, zero credits"
                         if not backend.live else
                         "see mh-receipts.jsonl and the spend ledger for charged amounts"),
        "budget": budget_status(model),
        "products": _product_entries(req.product_paths, members),
        "warnings": _cutout_warnings(members),
        "references": [Path(p).name for p in req.reference_paths],
        "reference_style": (asdict(ref_style) if ref_style else None),
        "instruction": req.instruction, "material": req.material,
        "knobs": req.knob_values, "aspect": aspect.value,
        "group_errors": group_errors,
        "variants": [asdict(v) for v in variants],
        "elapsed_s": round(elapsed, 2),
        "contact_sheet": "contact_sheet.png",
        "honesty": ("Scene-compose generates only the background and pastes "
                    "the source product cutout back; placement rescales the "
                    "cutout but the model never regenerates product pixels. "
                    "QA scores are automated proxies - human eye review "
                    "remains the acceptance gate per the brief."),
    }
    manifest_path = run_dir / "manifest.json"
    manifest["runtime_versions"] = runtime_versions()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _save_contact_sheet(run_dir, variants, members)
    return StudioResult(
        run_id=run_id, run_dir=run_dir, preset=preset.key,
        provider_requested=req.provider, variants=variants,
        group_errors=group_errors,
        reference_style=(asdict(ref_style) if ref_style else None),
        manifest_path=manifest_path, elapsed_s=elapsed)
