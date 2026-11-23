"""Pipeline orchestration. One run = every requested route x every variant seed,
with artifacts and a manifest written for audit. The full-run shape mirrors the
problem statement flow: source -> (cutout) -> candidates -> QA -> report.

Route 2 is always computable (CPU). Routes 1/3 need an editor backend; the CLI
supplies stub (tests), qwen (GPU) or magichour (mock/live). A backend raising on
a missing GPU fails only its own variants - the deterministic route still
delivers, and the manifest records the failure verbatim.
"""
from __future__ import annotations

import json
import hashlib
import time
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from . import compose, prompts, qa
from .fidelity_review import review_requirement
from .backends.base import EditorBackend
from .config import RunConfig
from .execution_kind import describe_execution
from .cutout import cutout, supplied_mask_cutout
from .relight import relight
from .source_provenance import identify_source
from .schemas import (EditRequest, QAReport, Route, RunResult, VariantResult)


def _run_dir(cfg: RunConfig, request: EditRequest) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Every attempt needs its own artifacts, even with the same timestamp.
    # Atomic creation also protects concurrent calls from reusing stale output.
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f"{request.source_path.stem}_{request.template.key}_{stamp}_",
        dir=cfg.out_root))


def run_pipeline(request: EditRequest, routes: list[Route], cfg: RunConfig,
                 editor: EditorBackend | None = None,
                 relight_enabled: bool = True, scene_fn=None,
                 snap_frame_cut: bool = True, glass_mode: bool = False,
                 scene_finish: bool = False, source_contract=None,
                 require_complete: bool = False,
                 allow_colour_conversion: bool = False, use_embedded_alpha: bool = False,
                 scene_accept=None, scene_max_attempts: int = 1) -> RunResult:
    prompts.validate_instruction(request.custom_instruction)
    from .pipeline_intake import load_generation_image
    if use_embedded_alpha and request.mask_path:
        raise ValueError("choose embedded alpha or separate mask, not both")
    if use_embedded_alpha and any(r not in (Route.CUTOUT_COMPOSE, Route.SCENE_COMPOSE) for r in routes):
        raise ValueError("embedded alpha is for source-product composition, not product regeneration")
    decoded = load_generation_image(request.source_path, allow_colour_conversion=allow_colour_conversion, use_embedded_alpha=use_embedded_alpha)
    source, intake_record = decoded[:2]
    embedded_alpha = decoded[2] if use_embedded_alpha else None
    if request.mask_path and intake_record["orientation"] != 1:
        raise ValueError("EXIF-oriented source with a separate mask requires prior normalized source/mask pairing; no automatic mask rotation")
    source_provenance = identify_source(request.source_path)
    ref_rgb = None
    ref_pil = None
    reference_intake = None
    if request.reference_path:
        import numpy as _np
        ref_pil, reference_intake = load_generation_image(request.reference_path, allow_colour_conversion=allow_colour_conversion)
        ref_rgb = _np.array(ref_pil)
    run_dir = _run_dir(cfg, request)
    if intake_record["transformations"]:
        source.save(run_dir / "normalized-source.png")
    if reference_intake and reference_intake["transformations"]:
        ref_pil.save(run_dir / "normalized-reference.png")
    aspect = request.effective_aspect

    # Cutout once per run: Routes 2/3 consume it, and QA uses its matte+bbox.
    # A cutout failure must degrade the run to recorded failures, never crash
    # the whole batch (benchmark honesty).
    inter_dir = run_dir / "intermediates" if cfg.save_intermediates else run_dir
    effective_mask_path = request.mask_path
    if embedded_alpha is not None:
        effective_mask_path = run_dir / "embedded-source-alpha.png"
        embedded_alpha.save(effective_mask_path)
        embedded_alpha.close()
    variants: list[VariantResult] = []
    product_transforms = {}
    co = None
    matte_np = None
    try:
        co = (supplied_mask_cutout(source, effective_mask_path, inter_dir)
              if effective_mask_path is not None else
              cutout(source, inter_dir, stem="product", backend="auto",
                     device=cfg.device.device))
        import numpy as np
        matte_np = np.array(Image.open(co.matte_path).convert("L"))
    except Exception as e:
        for route in routes:
            for seed in cfg.variant_seeds():
                report = QAReport(
                    source=str(request.source_path),
                    output=str(run_dir / f"{route.value}_seed{seed}.png"),
                    route=route.value, template=request.template.key, seed=seed,
                    critical_errors=[f"cutout failed: {type(e).__name__}: {e}"],
                    score_fidelity=None)
                variants.append(VariantResult(route, seed,
                                              report.output and Path(report.output),
                                              report, success=False))
        manifest = run_dir / "manifest.json"
        manifest.write_text(json.dumps({
            "source": str(request.source_path),
        "source_provenance": source_provenance,
        "image_intake": intake_record,
        "reference_intake": reference_intake,
            "template": asdict(request.template),
            "cutout_error": str(e),
            "variants": [{"route": v.route.value, "seed": v.seed,
                          "output": str(v.output_path), "success": False,
                          "backend": "cutout_failed", "provider_evidence": None,
                          "provider_call_status": "NOT_CALLED_CUTOUT_PREFLIGHT_FAILED",
                          "qa": json.loads(v.qa.to_json())} for v in variants],
        }, indent=2, default=str))
        return RunResult(request=request, variants=variants, run_dir=run_dir,
                         manifest_path=manifest)

    mask_provenance = "embedded_source_alpha" if use_embedded_alpha else "supplied_mask" if request.mask_path else None
    if request.mask_path:
        # Sidecar is artifact provenance only, never a quality/permission grant.
        sidecar = request.mask_path.parent / "mask-manifest.json"
        if sidecar.is_file():
            try:
                mp = json.loads(sidecar.read_text())
                if (mp.get("backend") in ("u2netp_experimental", "u2net_experimental", "seeded_grabcut_assisted")
                        and mp.get("alpha_sha256") == hashlib.sha256(request.mask_path.read_bytes()).hexdigest()
                        and mp.get("source_sha256") == hashlib.sha256(request.source_path.read_bytes()).hexdigest()):
                    mask_provenance = "estimated_" + mp["backend"]
            except (ValueError, OSError):
                pass
    from .source_contract import source_mask_preflight
    import numpy as np
    source_check = source_mask_preflight(np.array(source), matte_np, co.bbox,
        source_contract, overhead_scene=(request.template.key == 'flat_lay' or
            bool(getattr(scene_fn, 'flatlay_surface', False))),
        glass_mode=glass_mode, require_complete=require_complete)
    matched_pose = getattr(scene_fn,'camera_pose',None)
    if matched_pose and source_contract is not None and source_contract.pose not in ('unknown',matched_pose):
        source_check['blockers'].append('Declared source and requested scene camera poses disagree.')
        source_check['passed']=False
    glass_backdrop = None
    if glass_mode:
        B, clean = compose.estimate_clean_backdrop(np.array(source))
        if not clean:
            source_check['blockers'].append('Transmission requires a clean uniform source backdrop; refusing opaque fallback.')
        elif float(np.min(B)) < 128:
            source_check['blockers'].append('Transmission from a dark source backdrop is underdetermined; use a clean bright backdrop capture.')
        else:
            glass_backdrop = B
            source_check['warnings'].append('Single-image transmission is approximate: highlights and refracted geometry are not recovered; no exact RGB or material fidelity claim.')
        source_check['passed'] = not source_check['blockers']
    (inter_dir / 'source-mask-preflight.json').write_text(json.dumps(source_check, indent=2))
    prompt = prompts.build_prompt(request)
    neg = prompts.build_negative_prompt()

    for route in routes:
        for seed in cfg.variant_seeds():
            out_path = run_dir / f"{route.value}_seed{seed}.png"
            t0 = time.monotonic()
            editor_called = False
            prior_attempt_id = getattr(editor, '_attempt_id', None)
            try:
                if route in (Route.CUTOUT_COMPOSE, Route.SCENE_COMPOSE) and not source_check['passed']:
                    raise ValueError('Source compatibility preflight: ' + '; '.join(source_check['blockers']))
                if route in (Route.CUTOUT_COMPOSE, Route.HYBRID) and source_contract is not None and source_contract.material == 'transparent':
                    raise ValueError('Transparent source cannot use an opaque composite route, even when glass_mode is requested; use scene_compose with validated transmission.')
                if route is Route.CUTOUT_COMPOSE:
                    bg_tx = None
                    if relight_enabled:
                        # Grade the background only; product stays source-exact.
                        bg_tx = lambda im: relight(im, request.template.lighting,
                                                   prefer_gpu=False)[0]
                    img = compose.composite(co, request.template, aspect, seed,
                                            reflection="reflect" in request.template.scene.lower(),
                                            bg_transform=bg_tx, reference=ref_rgb)
                    img.save(out_path)
                    placed = _placed_region(co, aspect, seed, request.template.key)
                    transform = _product_transform_record(co.bbox, placed)
                    product_transforms[f"{route.value}:{seed}"] = transform
                    report = qa.evaluate(
                        source, out_path, route.value, request.template.key, seed,
                        src_crop_box=co.bbox, out_region=_placed_region(co, aspect, seed, request.template.key),
                        matte=matte_np,
                        notes=("route2: source RGB composited, no model repaint; " +
                               ("product resampled to template framing, native pixel-exactness NOT claimed"
                                if transform["product_resampled"] else
                                "native product dimensions retained; opaque RGB subject to output verification")))
                    # Pixel comparisons can't see a bad matte (a kept slab is
                    # source-exact); score the matte itself.
                    report.critical_errors.extend(qa.matte_sanity(matte_np, co.bbox, flag_frame_cut=True))
                    report.score_fidelity = qa.fidelity_score(report)
                    variants.append(VariantResult(route, seed, out_path, report,
                                                  {"cutout": co.rgba_path},
                                                  backend=co.backend,
                                                  seconds=round(time.monotonic() - t0, 2)))
                elif route is Route.SCENE_COMPOSE:
                    if not source_check['passed']:
                        raise ValueError('Source compatibility preflight: ' + '; '.join(source_check['blockers']))
                    # Route 2b: scene_fn(gray_canvas_rgb, prompt) -> rgb (IC-Light on GPU).
                    # Product pixels are pasted back source-exact. snap_frame_cut keeps
                    # frame-cut products from showing hard straight cuts (cover-fit or snap).
                    eff_scene_fn = scene_fn
                    if eff_scene_fn is None and ref_rgb is not None:
                        # Reference backdrop instead of a generated scene: cover-fit to
                        # whatever padded canvas aspect_safe_scene_call hands us.
                        eff_scene_fn = lambda g, pr: compose.cover_to_canvas(ref_rgb, (g.shape[1], g.shape[0]))
                    if eff_scene_fn is None:
                        raise RuntimeError("scene_compose needs a scene_fn (GPU relight) or a reference image")
                    gb, gnote = glass_backdrop, ""
                    img, diag = compose.composite_with_scene(
                        co, request.template, aspect, seed, eff_scene_fn,
                        shadow_style=request.template.shadow if scene_finish else None,
                        scene_auto_lift=scene_finish,
                        snap_frame_cut=snap_frame_cut, glass_backdrop=gb,
                        cache_dir=(inter_dir / f"scene_layers_seed{seed}") if cfg.save_intermediates else None,
                        scene_is_aligned=(scene_fn is None and ref_rgb is not None),
                        scene_accept=scene_accept, scene_max_attempts=scene_max_attempts)
                    img.save(out_path)
                    # Same-coordinate QA: compare the output region where the product was
                    # actually placed (diag["placement"]) against the source crop, instead of
                    # the whole canvas resized onto the source (invalid when the placed
                    # product is scaled/moved). Cover-fit placements compare the full source.
                    _pl = diag.get("placement")
                    _crop_box = (0, 0, source.width, source.height) if diag.get("cover_fit") else co.bbox
                    report = qa.evaluate(
                        source, out_path, route.value, request.template.key, seed,
                        src_crop_box=_crop_box, matte=matte_np, out_region=_pl,
                        notes=("route2b REFERENCE-BACKDROP (no generative scene model); "
                               if (scene_fn is None and ref_rgb is not None)
                               else "route2b: scene generated around source-exact product; ")
                              + f"ring diag {diag}; QA compares the actually placed region (same coordinates)"
                              + (" | GLASS MODE: product pixels modulated, not bit-exact" if gb is not None else "") + gnote)
                    # Structural matte checks (slabs, frame cut) stay on in every mode.
                    structural_flags = qa.matte_sanity(matte_np, co.bbox, flag_frame_cut=True)
                    if gb is None:
                        # A claimed exact paste is not permission to erase
                        # independently measured detail/colour-change warnings.
                        report.critical_errors.extend(structural_flags)
                    else:
                        # Glass modulation deliberately changes RGB, so its
                        # pixel-change warnings are not applicable. Structural
                        # warnings remain; the output stays explicitly unscored.
                        report.critical_errors = structural_flags
                    report.notes += f" | same-coordinate QA region={_pl} paste_bit_exact={diag.get('paste_bit_exact')} (pixel metrics use a resampled source baseline, not exact)"
                    if diag.get("paste_bit_exact") is False:
                        report.critical_errors.append("product paste NOT bit-exact at placed coordinates")
                    if gb is not None:
                        # Only the pixel-change metrics are inapplicable (pixels modulated by
                        # design). Say so explicitly and mark the output for eye review.
                        report.notes += (" | REVIEW NEEDED (glass mode): pixel-equality/colour/edge-retention "
                                         "metrics NOT APPLICABLE; matte checks kept; eye-check glass edges")
                    report.score_fidelity = None if gb is not None else qa.fidelity_score(report)
                    variants.append(VariantResult(route, seed, out_path, report,
                                                  {"cutout": co.rgba_path},
                                                  backend=co.backend,
                                                  seconds=round(time.monotonic() - t0, 2)))
                elif route is Route.PURE_PROMPT:
                    if editor is None:
                        raise RuntimeError("pure_prompt needs an editor backend")
                    editor_called = True
                    img = editor.edit(source, prompt, neg, seed, aspect, reference=ref_pil)
                    img.save(out_path)
                    report = qa.evaluate(
                        source, out_path, route.value, request.template.key, seed,
                        src_crop_box=co.bbox, matte=matte_np,
                        notes="route1: whole-image edit; alignment approximate")
                    variants.append(VariantResult(route, seed, out_path, report,
                                                  backend=getattr(editor, "name", "editor"),
                                                  seconds=round(time.monotonic() - t0, 2)))
                elif route is Route.HYBRID:
                    if editor is None:
                        raise RuntimeError("hybrid needs an editor backend")
                    base_img = compose.composite(co, request.template, aspect, seed,
                                                 reflection="reflect" in request.template.scene.lower())
                    harm = prompts.build_harmonize_prompt(request)
                    editor_called = True
                    img = editor.edit(base_img, harm, neg, seed, aspect, reference=ref_pil)
                    img.save(out_path)
                    report = qa.evaluate(
                        source, out_path, route.value, request.template.key, seed,
                        src_crop_box=co.bbox, out_region=_placed_region(co, aspect, seed, request.template.key),
                        matte=matte_np,
                        notes="route3: harmonize over correct composite; drift flagged")
                    report.critical_errors.extend(qa.matte_sanity(matte_np, co.bbox, flag_frame_cut=True))
                    report.score_fidelity = qa.fidelity_score(report)
                    variants.append(VariantResult(route, seed, out_path, report,
                                                  {"cutout": co.rgba_path},
                                                  backend=getattr(editor, "name", "editor"),
                                                  seconds=round(time.monotonic() - t0, 2)))
            except Exception as e:
                # Failed variant is recorded, not hidden (benchmark honesty).
                report = QAReport(source=str(request.source_path),
                                  output=str(out_path), route=route.value,
                                  template=request.template.key, seed=seed,
                                  critical_errors=[f"variant failed: {type(e).__name__}: {e}"],
                                  score_fidelity=None)
                variants.append(VariantResult(route, seed, out_path, report,
                                              seconds=round(time.monotonic() - t0, 2), success=False))

            if editor_called and callable(getattr(editor,'attempt_evidence',None)):
                evidence = editor.attempt_evidence()
                if isinstance(evidence,dict) and evidence.get('attempt_id') and evidence['attempt_id'] != prior_attempt_id:
                    evidence = dict(evidence)
                    evidence.update(pipeline_completed=variants[-1].success,
                        output_sha256=hashlib.sha256(out_path.read_bytes()).hexdigest() if out_path.is_file() else None,
                        pipeline_artifact_exists=out_path.is_file(),automatic_retry=False)
                    variants[-1].provider_evidence = evidence

    for variant in variants:
        variant.qa.notes += ' | SOURCE/MASK PREFLIGHT: ' + '; '.join(source_check['warnings'])
    if use_embedded_alpha:
        for variant in variants:
            variant.qa.notes += " | EMBEDDED SOURCE ALPHA: explicitly retained; no segmentation; mask semantics/material/publication unverified"
    if request.mask_path is not None:
        for variant in variants:
            variant.qa.notes += (" | ASSISTED SEEDED GRABCUT MASK: explicit correction regions/strokes; not automatic success or original GPU alpha; inspect edges/frame cuts"
                                 if mask_provenance == "estimated_seeded_grabcut_assisted" else
                                 f" | ESTIMATED {mask_provenance.replace('estimated_', '').upper()} MASK: new model estimate, not original GPU alpha; review missing objects/halos/transparency"
                                 if mask_provenance and mask_provenance.startswith("estimated_") else
                                 " | SUPPLIED MASK: explicit assisted alpha, not automatic segmentation; edges require review")
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "source": str(request.source_path),
        "source_provenance": source_provenance,
        "image_intake": intake_record,
        "reference_intake": reference_intake,
        "template": asdict(request.template),
        "custom_instruction": request.custom_instruction,
        "source_mask_preflight": source_check,
        "mask_path": str(effective_mask_path) if effective_mask_path else None,
        "mask_provenance": mask_provenance,
        "mask_sha256": hashlib.sha256(effective_mask_path.read_bytes()).hexdigest() if effective_mask_path else None,
        "aspect": aspect.value,
        "prompt": prompt if any(r is not Route.CUTOUT_COMPOSE for r in routes) else None,
        "cutout": asdict(co) | {"rgba_path": str(co.rgba_path),
                                "matte_path": str(co.matte_path)},
        "variants": [{
            "route": v.route.value, "seed": v.seed, "output": str(v.output_path),
            "backend": v.backend, "seconds": v.seconds,
            "credits_estimated": v.credits_estimated,
            "credits_estimated_semantics": "legacy_projection_not_actual_charge; use provider_evidence or unknown",
            "provider_evidence": v.provider_evidence,
            "success": v.success and v.output_path.is_file(),
            "artifact_exists": v.output_path.is_file(),
            "execution_kind": describe_execution(v.route.value, v.backend, scene_mock=getattr(scene_fn, "mock_scene", False) is True, reference=(scene_fn is None and ref_rgb is not None)),
            "fidelity_review": review_requirement(v.route.value, v.qa.critical_errors, output_exists=v.output_path.is_file(), completed=v.success),
            "product_transform": product_transforms.get(f"{v.route.value}:{v.seed}"),
            "qa": json.loads(v.qa.to_json()),
        } for v in variants],
    }, indent=2, default=str))
    return RunResult(request=request, variants=variants, run_dir=run_dir,
                     manifest_path=manifest)


def _placed_region(co, aspect, seed: int, template_key: str) -> tuple[int, int, int, int]:
    """Recompute where compose placed the crop for this variant seed. Uses the
    same placement_params/fit_product math as compose.composite so QA
    alignment and compositing can never drift apart silently."""
    cw, ch = aspect.size
    fill, jx, baseline = compose.placement_params(seed, template_key)
    x, y, w, h = co.bbox
    scale = min((ch * fill) / h, (cw * 0.9) / w, 1.0)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    px = max(0, min((cw - nw) // 2 + int(jx * cw), cw - nw))
    py = max(0, min(int(ch * baseline) - nh, ch - nh))
    return (px, py, nw, nh)


def _product_transform_record(source_bbox, placed_region):
    """Declared geometry only, not alpha truth or output pixel verification."""
    x,y,w,h=source_bbox
    px,py,nw,nh=placed_region
    return dict(schema='renderello-product-transform.v1',
        source_bbox=list(source_bbox),output_region=list(placed_region),
        source_crop_size=[w,h],placed_product_size=[nw,nh],
        product_resampled=(nw,nh)!=(w,h),
        native_rgb_exactness_claimed=False,
        alpha_semantics='NOT_VERIFIED',photographic_acceptance=False,
        note='No generative RGB repaint on cutout_compose. Resampling and fractional blending can change visible RGB. Unchanged dimensions alone do not certify native RGB.')
