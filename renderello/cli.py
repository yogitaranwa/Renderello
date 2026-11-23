"""CLI entry points. Defaults are safe: CPU device, mock Magic Hour, no live
calls, no GPU assumptions. Every flag that could spend or change behavior
outside the local machine is explicit opt-in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_benchmark
from .config import DeviceConfig, RunConfig
from .gallery import build_gallery
from .pipeline import run_pipeline
from .schemas import EditRequest, Route
from .templates import TEMPLATES, all_templates, get_template


def _fid(score):
    return "NOT_EVALUATED (review needed)" if score is None else f"{score}/5"


def _routes(text: str) -> list[Route]:
    names = [t.strip() for t in text.split(",") if t.strip()]
    out = []
    for n in names:
        try:
            out.append(Route(n))
        except ValueError:
            raise SystemExit(f"unknown route '{n}'. choices: "
                             f"{', '.join(r.value for r in Route)}")
    return out


def _editor(args, cfg: RunConfig):
    if args.editor == "stub":
        from .backends.base import StubEditor
        return StubEditor()
    if args.editor == "qwen":
        from .backends.qwen_local import QwenEditBackend
        return QwenEditBackend(device=cfg.device)
    if args.editor == "magichour":
        from .backends.magichour import MagicHourBackend
        live = bool(args.live)
        if live:
            print("WARNING: --live spends real Magic Hour credits. Proceeding only "
                  "because --live was passed explicitly.", file=sys.stderr)
        return MagicHourBackend(model=args.mh_model, live=live,
                                cost_log=cfg.out_root / "mh_cost_log.jsonl")
    return None


def _scene_fn(args):
    """Scene function for scene_compose. 'mock' = labeled gradient (no GPU).
    'iclight' = real IC-Light relight on CUDA (needs a GPU host + weights; never
    exercised on CPU)."""
    kind = getattr(args, "scene", "mock")
    if kind == "iclight":
        from .iclight_gpu import ICLight
        ic = ICLight(getattr(args, "iclight_weights", "models/iclight_sd15_fc.safetensors"))
        return (lambda gray, prompt: ic.relight(gray, prompt)[0]), ic
    from .webui import _mock_scene_fn
    return _mock_scene_fn, None


def main(argv: list[str] | None = None) -> int:
    # The candidate route owns its parser and lazy optional imports. Existing
    # commands remain usable without importing its color/matting dependencies.
    forwarded = list(sys.argv[1:] if argv is None else argv)
    research_commands = {
        'brief-economics': 'brief_economics',
        'provider-attempts': 'provider_attempts',
        'blind-review-return': 'blind_review_return',
        'native-region-matte': 'native_region_cli',
        'blind-review-pack': 'blind_pack_cli',
    }
    if forwarded and forwarded[0] in research_commands:
        from importlib import import_module
        import_module('.' + research_commands[forwarded[0]], __package__).main(forwarded[1:])
        return 0
    if forwarded[:1] == ['candidate-studio']:
        from .candidate_studio import main as candidate_main
        candidate_main(forwarded[1:])
        return 0
    p = argparse.ArgumentParser(prog="renderello",
                                description="AI product photography prototype pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in research_commands:
        sub.add_parser(name, help='offline evidence/research workflow; use '+name+' --help; no automatic quality acceptance')
    sub.add_parser('candidate-studio', help='source-bound multi-model masks, edge diagnostics, optional matting/contact/harmonization research; never automatic acceptance')

    run = sub.add_parser("run", help="run the pipeline on one source image")
    run.add_argument("--source", required=True, type=Path)
    run.add_argument("--allow-colour-conversion", action="store_true", help="explicit ICC-to-sRGB conversion for source/reference; original files retained, encoded RGB may change")
    run.add_argument("--templates", default="clean_studio",
                     help="comma list or 'all'")
    run.add_argument("--routes", default="cutout_compose")
    run.add_argument("--variants", type=int, default=2, choices=[1, 2, 3, 4])
    instructions=run.add_mutually_exclusive_group()
    instructions.add_argument("--instruction", default="")
    instructions.add_argument("--instruction-file", type=Path, help="UTF-8 multiline scene instruction; explicit 16000 character limit, no silent truncation")
    run.add_argument("--use-embedded-alpha", action="store_true", help="explicit uploaded source alpha for composition only; not mask/material verification")
    run.add_argument("--mask", type=Path, default=None, help="explicit source-sized grayscale alpha mask; manual assist, no auto segmentation")
    run.add_argument("--source-pose", choices=['unknown','front','elevated','overhead','suspended'], default='unknown')
    run.add_argument("--source-material", choices=['unknown','opaque','transparent'], default='unknown')
    run.add_argument("--source-framing", choices=['unknown','complete','cropped'], default='unknown')
    run.add_argument("--require-complete", action='store_true')
    run.add_argument("--reference", type=Path, default=None,
                     help="optional reference/backdrop photo (cover-fitted; PDF section 4 optional)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--editor", default="stub",
                     choices=["stub", "qwen", "magichour", "none"])
    run.add_argument("--mh-model", default="qwen-edit")
    run.add_argument("--mh-mock", action="store_true",
                     help="force Magic Hour mock (default unless --live)")
    run.add_argument("--live", action="store_true",
                     help="REAL Magic Hour calls - spends credits (owner approval required)")
    run.add_argument("--gpu", action="store_true", help="resolve CUDA device")
    run.add_argument("--no-relight", action="store_true")
    run.add_argument("--scene", choices=["mock", "iclight"], default="mock", help="scene_compose scene source: mock gradient (no GPU) or real IC-Light (GPU host only)")
    run.add_argument("--scene-finish", action="store_true", help="experimental scene-only contact shadow and dark-scene lift; CPU-tested, not real-GPU validated")
    run.add_argument("--glass-mode", action="store_true", help="scene_compose: clear-glass transmission mode (clean-backdrop sources only)")
    run.add_argument("--iclight-weights", default="models/iclight_sd15_fc.safetensors")
    run.add_argument("--out", type=Path, default=Path("runs"))

    cost = sub.add_parser("costs", help="cost projection from published MH list data (no spend)")
    cost.add_argument("--model", default="qwen-edit")
    cost.add_argument("--plan", default="creator_monthly")
    cost.add_argument("--credits", type=float, default=None,
                      help="override credits per generation (higher res / batch)")
    cost.add_argument("--images", type=int, default=1)

    cmp_ = sub.add_parser("mh-compare", help="S2: compare 2-3 Magic Hour editor models "
                          "on one source. Default prints a zero-spend plan; "
                          "--live SPENDS credits and needs owner approval")
    cmp_.add_argument("--source", type=Path)
    cmp_.add_argument("--models", default="flux-2-klein,qwen-edit,seedream-v4")
    cmp_.add_argument("--template", default="clean_studio")
    cmp_.add_argument("--plan", default="creator_monthly")
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--aspect", default="1:1", choices=["1:1", "4:5", "3:2"])
    cmp_.add_argument("--live", action="store_true",
                      help="SPEND Magic Hour credits (owner approval required)")
    cmp_.add_argument("--out", type=Path, default=Path("runs/mh-compare"))

    recover = sub.add_parser("mh-recover", help="read-only recovery of an existing MH project; no generation")
    recover.add_argument("--project-id", required=True)
    recover.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("scene-replay", help="CPU-only replay of saved exact layers, no model calls")
    replay.add_argument("--cache", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--shadow", choices=["soft", "hard", "none"])
    replay.add_argument("--lift", action="store_true")

    native = sub.add_parser("native-reframe", help="smaller source-sized alternate, no upscaling or model call")
    native.add_argument("--cache", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--fill", type=float, default=.76)

    ground = sub.add_parser("ground-replay", help="experimental explicit designed foreground, CPU-only")
    ground.add_argument("--cache", type=Path, required=True)
    ground.add_argument("--output", type=Path, required=True)
    ground.add_argument("--horizon", type=int, required=True)
    ground.add_argument("--color", type=int, nargs=3, required=True)

    bench = sub.add_parser("benchmark", help="run the acceptance benchmark")
    bench.add_argument("--manifest", required=True, type=Path)
    bench.add_argument("--allow-colour-conversion", action="store_true", help="explicit ICC-to-sRGB policy for all benchmark sources; preflight occurs before generation")
    bench.add_argument("--variants", type=int, default=2, choices=[1, 2, 3, 4])
    bench.add_argument("--editor", default="stub",
                       choices=["stub", "qwen", "magichour", "none"])
    bench.add_argument("--mh-model", default="qwen-edit")
    bench.add_argument("--live", action="store_true")
    bench.add_argument("--gpu", action="store_true")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--scene", default="mock", choices=["mock", "iclight"])
    bench.add_argument("--iclight-weights", default="models/iclight_sd15_fc.safetensors")
    bench.add_argument("--out", type=Path, default=Path("runs"))

    gal = sub.add_parser("gallery", help="build a private HTML gallery from run dirs")
    gal.add_argument("--runs", nargs="+", required=True, type=Path)
    gal.add_argument("--out", required=True, type=Path)

    summ = sub.add_parser("summarize", help="aggregate existing run manifests "
                          "through the benchmark gate math (no recompute)")
    summ.add_argument("--runs", nargs="+", required=True, type=Path)
    summ.add_argument("--out", required=True, type=Path)

    review_export = sub.add_parser("review-export", help="export blank seven-dimension ratings for existing outputs")
    review_export.add_argument("--runs", nargs="+", required=True, type=Path)
    review_export.add_argument("--out", required=True, type=Path)
    review_export.add_argument("--portable", action="store_true", help="write output paths relative to sheet folder")
    review_check = sub.add_parser("review-check", help="validate complete scores and exact output hashes; no acceptance claim")
    review_check.add_argument("--sheet", required=True, type=Path)

    bal = sub.add_parser("mh-balance", help="read-only Magic Hour credit balance")
    bal.add_argument("--json", action="store_true")

    matte = sub.add_parser("matte-refine", help="opt-in trimap matting/edge colors; no model downloads or acceptance claim")
    matte.add_argument("--source", type=Path, required=True)
    matte.add_argument("--alpha", type=Path, required=True)
    matte.add_argument("--trimap", type=Path, default=None)
    matte.add_argument("--method", choices=["cf","lkm","guided"], default="cf")
    matte.add_argument("--radius", type=int, default=3)
    matte.add_argument("--estimate-edge-colors", action="store_true", help="changes boundary RGB; preserves only definite foreground core exactly")
    matte.add_argument("--out", type=Path, required=True)

    matte_comp = sub.add_parser("matte-compose", help="compose a verified matting bundle on a declared overhead background")
    matte_comp.add_argument("--bundle", type=Path, required=True)
    matte_comp.add_argument("--background", type=Path, required=True, help="reviewed overhead background, not an upright scene")
    matte_comp.add_argument("--source-pose", choices=["overhead"], required=True)
    matte_comp.add_argument("--source-material", choices=["opaque"], required=True)
    matte_comp.add_argument("--source-framing", choices=["complete"], required=True)
    matte_comp.add_argument("--fill", type=float, default=.68)
    matte_comp.add_argument("--out", type=Path, required=True)

    semantic = sub.add_parser("semantic-mask", help="explicit isolated CPU IS-Net trial, not automatic quality acceptance")
    semantic.add_argument("--source", type=Path, required=True)
    semantic.add_argument("--out", type=Path, required=True)
    semantic.add_argument("--allow-download", action="store_true", help="allow official checksum-verified179MB IS-Net download inside Renderello")

    prompted = sub.add_parser("prompted-mask", help="explicit reviewed boxes/points with CPU Light HQ-SAM")
    for name in ("source", "out", "prompts", "checkpoint"):
        prompted.add_argument("--"+name,type=Path,required=True)

    pyramid = sub.add_parser("pyramid-mask", help="explicit CPU InSPyReNet fast experiment, not quality acceptance")
    for name in ("source", "out", "checkpoint"):
        pyramid.add_argument("--"+name,type=Path,required=True)

    bilateral = sub.add_parser("bilateral-mask", help="explicit CPU BiRefNet-lite trial; high memory use, not automatic acceptance")
    for name in ("source", "out", "checkpoint"):
        bilateral.add_argument("--"+name,type=Path,required=True)

    intent = sub.add_parser("intent-mask", help="explicit reviewed HQ-SAM intent intersected with existing automatic alpha")
    for name in ("source", "automatic", "reviewed", "out"):
        intent.add_argument("--"+name,type=Path,required=True)

    opaque = sub.add_parser("opaque-recover",help="explicit reviewed opaque silhouette to trimap matting; can replace missing saliency support, not automatic")
    for key in ('source','reviewed','out'):opaque.add_argument('--'+key,type=Path,required=True)
    opaque.add_argument('--reviewed-alpha-sha256',required=True)
    opaque.add_argument('--material',choices=['opaque','transparent','unknown'],required=True)
    opaque.add_argument('--framing',choices=['complete','cropped','unknown'],required=True)
    opaque.add_argument('--confirm-reviewed-silhouette',action='store_true')
    opaque.add_argument('--method',choices=['cf','lkm'],default='cf')
    opaque.add_argument('--radius',type=int,default=3)
    labels = sub.add_parser('label-retention',help='source-bound same-coordinate pixel/opacity checks, not OCR acceptance')
    for key in ('source','cutout','regions','out'):labels.add_argument('--'+key,type=Path,required=True)

    studio=sub.add_parser('native-studio',help='eight designed 2D studio candidates with native1:1product, not photorealistic template acceptance')
    studio.add_argument('--bundle',type=Path,required=True)
    studio.add_argument('--template',default='all')
    studio.add_argument('--out',type=Path,required=True)
    studio.add_argument('--regions',type=Path,help='optional exact source-bound print regions, checked in final scene after native translation')
    studio.add_argument('--source-pose',choices=['front','elevated','overhead','unknown'],required=True)
    studio.add_argument('--material',choices=['opaque','transparent','unknown'],required=True)
    studio.add_argument('--framing',choices=['complete','cropped','unknown'],required=True)

    refr=sub.add_parser('refractive-transfer',help='experimental clean-plate transfer with external reviewed flow/transmission, not a glass estimator')
    for field in ('source','plate','target','flow','transmission','support','spec','out'):
        refr.add_argument('--'+field,type=Path,required=True)
    refr.add_argument('--confirm-assumptions',action='store_true')

    plate=sub.add_parser('scene-plate',help='native product over an inspected 1024px RGB background with declared support geometry; not physical lighting or glass')
    for field in ('prepared','background','out'):
        plate.add_argument('--'+field,type=Path,required=True)
    placement=plate.add_mutually_exclusive_group(required=True)
    placement.add_argument('--baseline',type=int)
    placement.add_argument('--source-origin',type=int,nargs=2,metavar=('X','Y'),help='original normalized source origin on canvas; keeps source coordinates fixed across alpha changes')
    plate.add_argument('--center-x',type=int,default=512)
    plate.add_argument('--cast-offset',type=float,nargs=2,default=(0,0),metavar=('DX','DY'))
    plate.add_argument('--no-contact',action='store_true')
    plate.add_argument('--background-kind',choices=['photograph','generated','procedural','unspecified_inspected_background'],default='unspecified_inspected_background')
    plate.add_argument('--support-plane',type=Path,help='reviewed1024pxL support-plane mask; rejects unsupported feet and clips scene shadow')
    plate.add_argument('--contact-strength',type=float,default=1.)
    plate.add_argument('--shadow-model',choices=['legacy','height_soft_linear'],default='legacy')
    plate.add_argument('--shadow-softness',type=float,default=1.)
    plate.add_argument('--reviewed-contact-span',type=int,nargs=2,metavar=('LEFT','RIGHT'),help='explicit half-open original source columns for reviewed sloped-base contact; height_soft_linear only')
    plate.add_argument('--scene-exposure-ev',type=float,default=0.,help='background-only tone matching, half-stop maximum; not physical relighting')
    plate.add_argument('--scene-white-balance',type=float,nargs=3,default=(1.,1.,1.),metavar=('R','G','B'))

    args = p.parse_args(argv)
    if args.cmd == 'run':
        from .prompts import validate_instruction
        if args.instruction_file is not None:
            args.instruction=args.instruction_file.read_text(encoding='utf-8')
        args.instruction=validate_instruction(args.instruction)
    if args.cmd == 'scene-plate':
        from .scene_plate import compose_plate
        shadow_opts={} if args.shadow_model=='legacy' and args.shadow_softness==1. else dict(shadow_model=args.shadow_model,shadow_softness=args.shadow_softness)
        if args.reviewed_contact_span is not None:shadow_opts['reviewed_contact_span']=args.reviewed_contact_span
        if args.source_origin is not None:shadow_opts['source_origin']=args.source_origin
        result=compose_plate(args.prepared,args.background,args.out,baseline=args.baseline,
            center_x=args.center_x,cast_offset=args.cast_offset,contact=not args.no_contact,
            support_plane_path=args.support_plane,contact_strength=args.contact_strength,**shadow_opts,
            scene_exposure_ev=args.scene_exposure_ev,scene_white_balance=args.scene_white_balance,background_kind=args.background_kind)
        print(json.dumps(result,indent=2));return 0

    if args.cmd == 'refractive-transfer':
        from .refractive_transfer import transfer_files
        print(json.dumps(transfer_files(args.source,args.plate,args.target,args.flow,args.transmission,args.support,args.spec,args.out,confirm_assumptions=args.confirm_assumptions),indent=2));return 0

    if args.cmd == 'native-studio':
        from .native_studio import compose_native,STYLES
        keys=list(STYLES) if args.template=='all' else args.template.split(',')
        if len(set(keys))!=len(keys) or any(k not in STYLES for k in keys):raise ValueError('invalid or duplicate native studio template')
        if args.out.exists():raise ValueError('output exists; preserve earlier candidates')
        print(json.dumps([compose_native(args.bundle,k,args.out/k,source_pose=args.source_pose,material=args.material,framing=args.framing,regions_path=args.regions) for k in keys],indent=2));return 0
    if args.cmd == 'opaque-recover':
        from .opaque_recovery import recover_file
        print(json.dumps(recover_file(args.source,args.reviewed,args.out,reviewed_alpha_sha256=args.reviewed_alpha_sha256,material=args.material,framing=args.framing,confirm_reviewed_silhouette=args.confirm_reviewed_silhouette,method=args.method,radius=args.radius),indent=2));return 0
    if args.cmd == 'label-retention':
        from .label_retention import check_file
        print(json.dumps(check_file(args.source,args.cutout,args.regions,args.out),indent=2));return 0
    if args.cmd == "intent-mask":
        from .intent_mask import gate_file
        print(json.dumps(gate_file(args.source,args.automatic,args.reviewed,args.out),indent=2))
        return 0
    if args.cmd == "bilateral-mask":
        from .bilateral_mask import guarded_file
        print(json.dumps(guarded_file(args.source,args.out,args.checkpoint),indent=2))
        return 0
    if args.cmd == "pyramid-mask":
        from .pyramid_mask import guarded_file
        print(json.dumps(guarded_file(args.source,args.out,args.checkpoint),indent=2))
        return 0
    if args.cmd == "prompted-mask":
        from .prompted_mask import guarded_file
        print(json.dumps(guarded_file(args.source,args.out,args.prompts,args.checkpoint),indent=2))
        return 0
    if args.cmd == "semantic-mask":
        from .semantic_mask import guarded_file
        print(json.dumps(guarded_file(args.source,args.out,args.allow_download),indent=2))
        return 0
    if args.cmd == "matte-compose":
        from .matting import compose_bundle
        print(json.dumps(compose_bundle(args.bundle,args.background,args.out,source_pose=args.source_pose,
              material=args.source_material,framing=args.source_framing,fill=args.fill),indent=2))
        return 0
    if args.cmd == "matte-refine":
        from .matting import guarded_file
        print(json.dumps(guarded_file(args.source,args.alpha,args.out,trimap_path=args.trimap,
              method=args.method,radius=args.radius,edge_colors=args.estimate_edge_colors),indent=2))
        return 0
    if args.cmd == "ground-replay":
        from .ground_plane import replay_ground_plane
        print(json.dumps(replay_ground_plane(args.cache, output=args.output,
              horizon=args.horizon, color=args.color), indent=2))
        return 0

    if args.cmd == "review-export":
        from .review import export_sheet
        print(export_sheet(args.runs, args.out, portable=args.portable))
        print("Blank ratings only. No scores inferred; missing image files are not reviewable.")
        return 0
    if args.cmd == "review-check":
        from .review import load_scores
        records = load_scores(args.sheet)
        print(f"Validated {len(records)} score records against image bytes. Rater labels are self-reported, not authenticated.")
        print("Acceptance NOT_EVALUATED. Full source/template coverage and specialist comparison remain separate.")
        return 0
    if args.cmd == "mh-balance":
        from .backends.magichour import MagicHourBackend
        info = MagicHourBackend().balance()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"credits: {info.get('credits')} (tier: {info.get('tier')})")
        return 0

    if args.cmd == "mh-recover":
        from .backends.magichour import MagicHourBackend
        backend = MagicHourBackend()
        image = backend.resume(args.project_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        args.output.with_suffix(".json").write_text(json.dumps(backend.last_result, indent=2))
        print(f"Recovered existing project {args.project_id} -> {args.output}; no generation submitted")
        return 0
    if args.cmd == "native-reframe":
        from .native_reframe import native_reframe
        image, diagnostic = native_reframe(args.cache, fill=args.fill)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        args.output.with_suffix(".json").write_text(json.dumps(diagnostic, indent=2))
        print(json.dumps(diagnostic))
        return 0
    if args.cmd == "scene-replay":
        from .compose import replay_scene
        overrides = {}
        if args.shadow:
            overrides["shadow_style"] = None if args.shadow == "none" else args.shadow
        if args.lift:
            overrides["scene_auto_lift"] = True
        image, diagnostic = replay_scene(args.cache, **overrides)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(f"CPU replay -> {args.output}: {diagnostic}; visual quality NOT_EVALUATED")
        return 0
    if args.cmd == "costs":
        from . import costs as _c
        rate = _c.effective_usd_per_credit(args.plan)
        per = _c.projected_usd_per_output(args.model, args.plan,
                                          images_per_generation=args.images,
                                          credits_per_generation=args.credits)
        print(f"PROJECTION (published list data, no spend): {args.model} under {args.plan}")
        print(f"  included-credit rate: ${rate:.6f}/credit (bulk average, not marginal)")
        print(f"  projected cost per output: ${per:.6f}")
        print("  gross margin = sale price - this cost; sale price is your input, not assumed")
        return 0
    if args.cmd == "mh-compare":
        from .mhcompare import plan_compare, run_compare
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not args.live:
            plan = plan_compare(models, plan=args.plan)
            print("PLAN ONLY - no credits spent (pass --live to execute; needs owner approval)")
            for r in plan["models"]:
                print(f"  {r['model']:<18} {r['credits_listed_from']:>4} credits "
                      f"(tier {r['tier_required']:<7}) ~${r['projected_usd_per_image']:.4f}/img")
            print(f"  total: {plan['total_credits_listed']} credits, "
                  f"~${plan['total_projected_usd']:.4f} (bulk included-credit rate, projection)")
            return 0
        from .templates import get_template as _get_template
        from .schemas import Aspect as _A
        if args.source is None:
            p.error("--source is required for --live")
        manifest = run_compare(args.source, models, _get_template(args.template),
                               args.seed, _A(args.aspect), args.out, live=True)
        print(f"LIVE comparison complete: {len(manifest['results'])} models -> {args.out}")
        for r in manifest["results"]:
            print(f"  {r['model']:<18} {r['latency_s_measured']:>7.1f}s  {r['output']}")
        return 0
    cfg = RunConfig(seed=getattr(args, "seed", 0),
                    variants=getattr(args, "variants", 2),
                    out_root=args.out,
                    device=DeviceConfig.resolve(prefer_gpu=getattr(args, "gpu", False)))

    if args.cmd == "run":
        editor = _editor(args, cfg)
        templates = all_templates() if args.templates == "all" else \
            [get_template(k) for k in args.templates.split(",")]
        routes = _routes(args.routes)
        scene_fn, _ic = None, None
        if Route.SCENE_COMPOSE in routes:
            scene_fn, _ic = _scene_fn(args)
        if any(r in (Route.PURE_PROMPT, Route.HYBRID) for r in routes) and editor is None:
            raise SystemExit("pure_prompt/hybrid need --editor stub|qwen|magichour")
        from .source_contract import SourceContract
        contract = SourceContract(args.source_pose,args.source_material,args.source_framing)
        for tpl in templates:
            req = EditRequest(source_path=args.source, template=tpl,
                              custom_instruction=args.instruction, seed=args.seed,
                              reference_path=args.reference, mask_path=args.mask)
            result = run_pipeline(req, routes, cfg, editor=editor,
                                  relight_enabled=not args.no_relight, scene_fn=scene_fn, glass_mode=args.glass_mode, scene_finish=args.scene_finish, source_contract=contract, require_complete=args.require_complete, allow_colour_conversion=args.allow_colour_conversion, use_embedded_alpha=args.use_embedded_alpha)
            print(f"[{tpl.key}] {len(result.variants)} variants -> {result.run_dir}")
            for v in result.variants:
                flag = " FLAGS" if v.qa.critical_errors else ""
                print(f"  {v.route.value} seed{v.seed}: fidelity "
                      f"{_fid(v.qa.score_fidelity)}{flag} -> {v.output_path}")
        if _ic is not None:
            _ic.close()
        return 0

    if args.cmd == "benchmark":
        editor = _editor(args, cfg)
        scene_fn, _ic, kind = None, None, ""
        if "scene_compose" in json.loads(args.manifest.read_text()).get("routes", []):
            scene_fn, _ic = _scene_fn(args)
            kind = args.scene
        csv_path = run_benchmark(args.manifest, cfg, editor=editor, scene_fn=scene_fn, scene_kind=kind, allow_colour_conversion=args.allow_colour_conversion)
        if _ic is not None:
            _ic.close()
        print(f"benchmark scores -> {csv_path}")
        return 0

    if args.cmd == "gallery":
        out = build_gallery(args.runs, args.out)
        print(f"gallery -> {out}")
        return 0

    if args.cmd == "summarize":
        from .benchmark import summarize_runs
        csv_path = summarize_runs(args.runs, args.out)
        print(f"gate summary -> {csv_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


