"""Minimal local web UI for the problem statement's working flow (section 4):
upload real product photo -> select template WITH VISUAL PREVIEW -> optional
multiline custom instruction -> generate 2-4 alternatives -> compare/download or
edit/regenerate, with square/portrait/landscape output where the pipeline
permits. Stdlib only (no new deps), binds 127.0.0.1, serves run artifacts
from the out dir. This is a prototype UI for the owner's machine, not a
hosted product.

Honesty labels: result scores are AUTOMATED STRUCTURAL metrics (pixel-level
fidelity checks), not human ratings and not publish-ready judgments. Template
previews are demo renders on a synthetic placeholder product until the real
20+ source-image benchmark supplies genuine cover examples.

Run:  python -m renderello.webui --port 8000 --out runs/webui
"""
from __future__ import annotations

import zipfile
import argparse
import html
import io
import json
import re
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .backends.base import StubEditor
from .config import RunConfig
from .pipeline import run_pipeline
from .prompts import validate_instruction
from .gallery import _source_caption
from .schemas import Aspect, EditRequest, Route
from .templates import all_templates, get_template
from .source_contract import SourceContract

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Renderello</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;
padding:0 14px;color:#1c1c1e;overflow-x:hidden}}
label{{display:block;margin-top:12px;font-weight:600}}
select,input[type=text],textarea{{width:100%;padding:8px;margin-top:4px}}
textarea{{min-height:130px;resize:vertical;font:inherit;line-height:1.45}}
.tplgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
gap:10px;margin-top:6px}}
.tpl{{border:2px solid #ccc;border-radius:8px;padding:8px;cursor:pointer;
font-weight:400;font-size:13px;background:#fff}}
.tpl input{{margin-right:6px}}
.tpl img{{width:100%;border-radius:5px;display:block;margin-bottom:6px;
background:#f2f2f4;min-height:60px}}
.tpl.sel{{border-color:#0a6cff}}
.card{{border:1px solid #ccc;border-radius:8px;padding:14px;margin-top:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
gap:14px}}
.shot img{{width:100%;border-radius:6px;display:block;border:1px solid #e2e2e4}}
.meta{{overflow-wrap:anywhere;font-size:12px;color:#555;white-space:pre-wrap;margin-top:6px}}
.bad{{color:#a00}}
a.dl{{display:inline-block;margin-top:8px;padding:6px 12px;background:#0a6cff;
color:#fff;border-radius:6px;text-decoration:none;font-size:13px}}
.note{{font-size:12px;color:#777;margin-top:4px}}
button{{padding:10px 22px;font-size:15px;background:#0a6cff;color:#fff;
border:0;border-radius:8px;cursor:pointer}}
</style>
<script>
function pick(id){{document.querySelectorAll('.tpl').forEach(e=>e.classList.remove('sel'));
document.getElementById('l'+id).classList.add('sel');}}
</script></head><body>
<h1>Renderello / Shotelier - product photography prototype</h1>
<p class="meta">{backend_status}</p>
<p><a href="/mask-prompts">Prepare reviewed mask boxes / points (optional)</a></p>
<form method="post" action="/run" enctype="multipart/form-data">
<label>Product photo <input type="file" name="photo" accept="image/*" required></label>
<label><input type="checkbox" name="allow_colour_conversion" value="yes">Allow embedded ICC profile conversion to sRGB (source and reference). Original files are retained. Normalized RGB may differ; no colour-accuracy guarantee.</label>
<label>Template (demo previews on a synthetic placeholder product; real cover
examples come from the benchmark)</label>
<div class="tplgrid">{options}</div>
<label>Custom instruction (scene/lighting only, optional)
<textarea name="instruction" maxlength="16000" rows="6"
 placeholder="Describe the surface, scene, light, camera and placement in detail"></textarea></label>
<p class="note">Up to 16,000 characters. Full text is retained without silent cuts. Image models may still ignore details. Procedural and mock routes do not interpret every clause.</p>
<label><input type="checkbox" name="use_embedded_alpha" value="yes">Use the uploaded source's existing alpha for cutout/scene composition. No segmentation model; alpha quality and material semantics remain unverified. Cannot combine with a separate alpha or RGBA result.</label>
<label>Optional alpha mask (8-bit grayscale PNG, same dimensions as source; white product, black background)<input type="file" name="mask" accept="image/png"></label>
<label>Optional source-RGB-locked RGBA result (PNG)
<input type="file" name="rgba_result" accept="image/png"></label>
<p class="note">Use with a normalized RGB PNG source and cutout/scene compose only. Every imported RGB pixel must match the source. Alpha is retained for review, not certified. Do not supply a separate alpha mask as well. No model is called for this import.</p>
<label>Reference/backdrop photo (optional; used as the scene backdrop,
 cover-fitted to the chosen aspect)
<input type="file" name="reference" accept="image/*"></label>
<label>Output aspect
<select name="aspect">
<option value="">Template default</option>
<option value="1:1">Square 1:1</option>
<option value="4:5">Portrait 4:5</option>
<option value="3:2">Landscape 3:2</option></select></label>
<label>Alternatives <select name="variants">
<option>2</option><option>3</option><option>4</option></select></label>
<fieldset><legend>Source review (your declarations, not automatic image analysis)</legend>
<label>Camera / pose <select name="source_pose"><option value="unknown">Not reviewed</option><option value="front">Front / low view</option><option value="elevated">Elevated view</option><option value="overhead">Overhead</option><option value="suspended">Suspended / floating</option></select></label>
<label>Material <select name="source_material"><option value="unknown">Not reviewed</option><option value="opaque">Opaque</option><option value="transparent">Transparent / translucent</option></select></label>
<label>Product framing <select name="source_framing"><option value="unknown">Not reviewed</option><option value="complete">Whole product present</option><option value="cropped">Cropped / missing parts</option></select></label>
<label><input type="checkbox" name="require_complete" value="yes"> Require a complete product (block detected frame cuts)</label>
<p class="note">Source-exact paste cannot change the camera angle or restore missing parts. Transparent products need a separate transmission workflow. Genuine fine-print preservation is not verified.</p></fieldset>
<label>Route <select name="route">
<option value="cutout_compose">cutout + compose (source pixels; mask review required)</option>
<option value="pure_prompt">pure prompt (stub editor unless GPU build)</option>
<option value="hybrid">hybrid</option>
<option value="scene_compose">scene compose (MOCK scene unless GPU relight is plugged in)</option></select></label>
<p><button type="submit">Generate</button></p>
</form>{result}</body></html>"""

_RESULT_HEAD = ('<div class="card"><h2>Results</h2>'
                '<p class="note">Prototype outputs. Scores below are automated '
                'structural fidelity metrics only - not human ratings and not '
                'publish-ready judgments.</p><div class="grid">')

_RERUN = """</div></div>
<div class="card"><h2>Edit / regenerate</h2>
<form method="post" action="/rerun">
<input type="hidden" name="job" value="{job}">
<label><input type="checkbox" name="use_embedded_alpha" value="yes">Use source existing alpha (composition only; mask/material unverified).</label>
<label><input type="checkbox" name="allow_colour_conversion" value="yes">Allow embedded ICC profile conversion to sRGB (source/reference). Original files retained; normalized RGB may change.</label>
<label>Template <select name="template">{options}</select></label>
<label>Custom instruction
<textarea name="instruction" maxlength="16000" rows="6">{instruction}</textarea></label>
<p class="note">Up to 16,000 characters, retained without silent cuts. Model obedience is not guaranteed.</p>
<label>Output aspect
<select name="aspect">
<option value="">Template default</option>
<option value="1:1"{a11}>Square 1:1</option>
<option value="4:5"{a45}>Portrait 4:5</option>
<option value="3:2"{a32}>Landscape 3:2</option></select></label>
<label>Alternatives <select name="variants">
<option>2</option><option>3</option><option>4</option></select></label>
<fieldset><legend>Source review (your declarations, not automatic image analysis)</legend>
<label>Camera / pose <select name="source_pose"><option value="unknown">Not reviewed</option><option value="front">Front / low view</option><option value="elevated">Elevated view</option><option value="overhead">Overhead</option><option value="suspended">Suspended / floating</option></select></label>
<label>Material <select name="source_material"><option value="unknown">Not reviewed</option><option value="opaque">Opaque</option><option value="transparent">Transparent / translucent</option></select></label>
<label>Product framing <select name="source_framing"><option value="unknown">Not reviewed</option><option value="complete">Whole product present</option><option value="cropped">Cropped / missing parts</option></select></label>
<label><input type="checkbox" name="require_complete" value="yes"> Require a complete product (block detected frame cuts)</label>
<p class="note">Source-exact paste cannot change the camera angle or restore missing parts. Transparent products need a separate transmission workflow. Genuine fine-print preservation is not verified.</p></fieldset>
<label>Route <select name="route">
<option value="cutout_compose">cutout + compose (source pixels; mask review required)</option>
<option value="pure_prompt">pure prompt (stub editor unless GPU build)</option>
<option value="hybrid">hybrid</option>
<option value="scene_compose">scene compose (MOCK scene unless GPU relight is plugged in)</option></select></label>
<p><button type="submit">Regenerate with new seeds</button></p>
</form></div>"""


def _parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str, bytes]]:
    """Small strict multipart parser (stdlib cgi is deprecated). Returns
    field -> (filename or '', data)."""
    from email.parser import BytesParser
    from email.policy import default
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("multipart/form-data required")
    message = BytesParser(policy=default).parsebytes(
        ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
    if not message.is_multipart():
        raise ValueError("multipart boundary missing or invalid")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = (part.get_filename() or "", part.get_payload(decode=True) or b"")
    return fields


def _template_cards(selected: str = "") -> str:
    cards = []
    for i, t in enumerate(all_templates()):
        sel = " sel" if (t.key == selected or (not selected and i == 0)) else ""
        chk = " checked" if (t.key == selected or (not selected and i == 0)) else ""
        cards.append(
            f'<label class="tpl{sel}" id="l{t.key}">'
            f'<img src="/preview/{t.key}.png" loading="lazy" alt="" '
            f'onerror="this.style.display=\'none\'">'
            f'<input type="radio" name="template" value="{t.key}"{chk} '
            f'onclick="pick(\'{t.key}\')">{html.escape(t.name)}'
            f'<div class="note">{html.escape(t.category)} - {t.aspect.value}</div>'
            f'</label>')
    return "".join(cards)


def _aspect_from(value: str) -> Aspect | None:
    return Aspect(value) if value in (a.value for a in Aspect) else None


def _render_previews(out_root: Path) -> None:
    """Render one demo preview per template on a synthetic placeholder product.
    Prepared before server start with known alpha, no segmentation/model call.
    These are NOT the benchmark cover examples."""
    from PIL import Image, ImageDraw
    from .cutout import _save_cutout
    import numpy as np
    from .compose import composite
    prev = out_root / "_previews"
    prev.mkdir(parents=True, exist_ok=True)
    if all((prev / f"{t.key}.png").exists() for t in all_templates()):
        return
    img = Image.new("RGB", (480, 480), (170, 172, 174))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 384, 480, 480], fill=(150, 152, 154))
    d.rectangle([190, 130, 290, 380], fill=(36, 130, 140))
    d.rectangle([210, 96, 270, 130], fill=(28, 28, 30))
    d.rectangle([200, 210, 280, 330], fill=(246, 246, 244), outline=(60, 60, 60))
    src = prev / "_demo_source.png"
    img.save(src)
    # This is our synthetic placeholder, so its exact alpha is known. Do not
    # launch a model or GrabCut for preview generation or race the first page.
    alpha = Image.new("L", img.size, 0)
    ad = ImageDraw.Draw(alpha)
    ad.rectangle([190,130,290,380],fill=255)
    ad.rectangle([210,96,270,130],fill=255)
    co = _save_cutout(np.array(img), np.array(alpha), prev / "_cutout", "demo", "synthetic_known_alpha")
    for t in all_templates():
        out = prev / f"{t.key}.png"
        if out.exists():
            continue
        try:
            composite(co, t, t.aspect, seed=7,
                      reflection="reflect" in t.scene.lower()).save(out)
        except Exception:
            pass  # missing preview is cosmetic only


def _mock_scene_fn(gray, prompt):
    """MOCK scene for the demo UI (labeled): vertical warm gradient. A GPU build
    replaces this with IC-Light (iclight_gpu.ICLight.relight). Never real relighting."""
    import numpy as np
    h, w = gray.shape[:2]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top, bot = np.array([205, 200, 192], np.float32), np.array([120, 105, 92], np.float32)
    return (top * (1 - y) + bot * y).repeat(w, axis=1).astype(np.uint8)


_mock_scene_fn.mock_scene = True

class _Handler(BaseHTTPRequestHandler):
    out_root: Path = Path("runs/webui")

    def _send(self, code: int, body: bytes, ctype: str = "text/html"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, rel: str):
        root = self.out_root.resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            self._send(403, b"forbidden", "text/plain")
            return
        if target.is_file():
            ctype = {".png": "image/png", ".jpg": "image/jpeg",
                     ".html": "text/html"}.get(target.suffix, "text/plain")
            self._send(200, target.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/reviewed-status":
            checkpoint=getattr(self,"reviewed_checkpoint",None)
            self._send(200,json.dumps({"configured": bool(checkpoint and checkpoint.is_file())}).encode(),"application/json")
            return
        if path == "/mask-prompts":
            from .prompt_editor import PAGE
            self._send(200,PAGE.encode())
            return
        if path == "/":
            self._send(200, _PAGE.format(options=_template_cards(),
                                         result="", backend_status=html.escape(getattr(self,"backend_status","Offline stub editor; no live generation"))).encode())
            return
        if path.startswith("/preview/"):
            key = re.sub(r"[^a-z0-9_]", "", path[len("/preview/"):].removesuffix(".png"))
            self._serve_file(f"_previews/{key}.png")
            return
        if path.startswith("/file/"):
            self._serve_file(path[len("/file/"):])
            return
        self._send(404, b"not found", "text/plain")

    def _run_job(self, src: Path, template_key: str, instruction: str,
                 aspect: Aspect | None, variants: int, route: Route,
                 ref_upload: tuple[str, bytes] | None = None,
                 mask_upload: tuple[str, bytes] | None = None,
                 source_contract: SourceContract | None = None, require_complete: bool = False,
                 rgba_upload: tuple[str, bytes] | None = None, allow_colour_conversion: bool = False, use_embedded_alpha: bool = False):
        if use_embedded_alpha and (mask_upload or rgba_upload):
            raise ValueError("choose embedded alpha, separate mask or RGBA result, not a combination")
        if variants not in (2,3,4):
            raise ValueError("alternatives must be 2, 3 or 4")
        job = self.out_root / uuid.uuid4().hex[:12]
        job.mkdir(parents=True, exist_ok=True)
        dst = job / f"source{src.suffix}"
        dst.write_bytes(src.read_bytes())
        contract = source_contract or SourceContract()
        (job / "source-review.json").write_text(json.dumps({"pose":contract.pose,"material":contract.material,
            "framing":contract.framing,"require_complete":require_complete}))
        tpl = get_template(template_key)
        ref_dst = None
        if ref_upload and ref_upload[1]:
            ref_dst = job / f"reference{Path(ref_upload[0] or 'ref.png').suffix or '.png'}"
            ref_dst.write_bytes(ref_upload[1])
        mask_dst = None
        if mask_upload and mask_upload[1]:
            mask_dst = job / "mask.png"
            mask_dst.write_bytes(mask_upload[1])
        if rgba_upload and rgba_upload[1]:
            if mask_upload and mask_upload[1]:raise ValueError("Choose RGBA result or separate alpha mask, not both")
            if route not in (Route.CUTOUT_COMPOSE,Route.SCENE_COMPOSE):raise ValueError("RGBA import is only for cutout/scene compose, not product regeneration")
            from .reviewed_edge_cli import read_image,sha
            from .source_locked_rgba_import import load_source_locked_rgba
            from PIL import Image
            # This opt-in path requires already normalized native RGB PNG,
            # avoiding a second normalization convention in the legacy pipeline.
            rgb,source_record=read_image(dst,sha(dst.read_bytes()),'source')
            imported=job/'external-rgba.png';imported.write_bytes(rgba_upload[1])
            alpha,evidence=load_source_locked_rgba(imported,rgb,sha(rgba_upload[1]))
            mask_dst=job/'mask.png';Image.fromarray(alpha).save(mask_dst)
            evidence.update(source_file_sha256=source_record['file_sha256'],source_normalized_by_import=False,alpha_semantics='NOT_VERIFIED',publication_status='NOT_EVALUATED')
            (job/'rgba-import.json').write_text(json.dumps(evidence,indent=2))
        cfg = RunConfig(seed=secrets.randbelow(10_000), variants=variants,
                        out_root=job)
        if getattr(self,'live_editor',False) and route is Route.HYBRID:
            raise ValueError('Magic Hour mask-constrained hybrid is unsupported; use pure_prompt or an explicit local compose route')
        editor = self.editor_factory(job) if route in (Route.PURE_PROMPT, Route.HYBRID) else None
        # An uploaded reference is the user's chosen backdrop: prefer it over the
        # mock scene so scene_compose composites onto THEIR photo (GPU-free path).
        scene_fn = _mock_scene_fn if (route is Route.SCENE_COMPOSE and ref_dst is None) else None
        result = run_pipeline(EditRequest(source_path=dst, template=tpl,
                                          custom_instruction=instruction,
                                          aspect=aspect, reference_path=ref_dst, mask_path=mask_dst),
                              [route], cfg, editor=editor, scene_fn=scene_fn,
                              source_contract=contract,require_complete=require_complete,allow_colour_conversion=allow_colour_conversion,use_embedded_alpha=use_embedded_alpha)
        return job, dst, result

    def _results_page(self, job: Path, src: Path, result, template_key: str,
                      instruction: str, aspect: Aspect | None) -> bytes:
        cards = [_RESULT_HEAD]
        import_record=job/'rgba-import.json'
        if import_record.is_file():
            cards.append('<p class="meta" style="grid-column:1/-1">VERIFIED SOURCE-RGB RGBA IMPORT: native PNG RGB matches every source pixel. Alpha semantics and publication quality are NOT VERIFIED. The same saved RGBA is rechecked on regenerate. No model is called by import.</p>')
        manifest = json.loads(result.manifest_path.read_text())
        intake = manifest.get("image_intake", {})
        if intake:
            import html as _html
            if intake.get("embedded_alpha_used"):
                cards.append('<p class="meta" style="grid-column:1/-1">EMBEDDED SOURCE ALPHA: retained from the upload after orientation normalization. No segmentation model was called. Mask quality and material semantics are NOT VERIFIED.</p>')
            text = "IMAGE INTAKE: " + ", ".join(intake.get("transformations", [])) if intake.get("transformations") else "IMAGE INTAKE: unprofiled decoded RGB assumed sRGB, not colour-accuracy verified."
            cards.append('<p class="meta" style="grid-column:1/-1">' + _html.escape(text + " Product comparisons use normalized RGB, not original encoded colour values.") + "</p>")
        check = manifest.get("source_mask_preflight", {})
        if check:
            review_text = "SOURCE REVIEW: " + json.dumps(check.get("declaration", {})) + "\n"
            review_text += "\n".join(check.get("blockers", []) + check.get("warnings", []))
            review_text += "\nMask semantic accuracy: NOT VERIFIED. Publication readiness: NOT EVALUATED."
            cards.append('<div class="meta bad" style="grid-column:1/-1">' + html.escape(review_text) + '</div>')
        if any(v.get('product_transform',{}).get('product_resampled') for v in manifest.get('variants',[]) if isinstance(v.get('product_transform'),dict)):
            cards.append('<p class="meta" style="grid-column:1/-1">PRODUCT RESAMPLING: one or more alternatives resize the product to template framing. Resampling and alpha blending can change visible RGB. Import verification, when supplied, does not certify native pixel-exact final outputs. Inspect fine print in downloads.</p>')
        cards.append(f'<div class="shot"><img src="/file/{job.name}/{src.name}">'
                     f'<div class="meta">{html.escape(_source_caption(manifest))}</div></div>')
        for v in result.variants:
            if not v.success or not v.output_path.is_file():
                flags = "; ".join(v.qa.critical_errors) or "Output file is missing"
                partial = v.output_path.is_file()
                text = (("FAILED / PARTIAL ARTIFACT - NOT A COMPLETED OUTPUT\n" if partial else "FAILED / NOT GENERATED\n") + f"{v.route.value} seed {v.seed}\n" +
                        ("Artifact retained for debugging. No quality rating.\n" if partial else "No image output and no quality rating.\n") + flags)
                cards.append(f'<div class="shot"><div class="meta bad">{html.escape(text)}</div></div>')
                continue
            from .execution_kind import describe_execution
            recorded = next((row.get("execution_kind") for row in manifest.get("variants",[]) if row.get("route")==v.route.value and row.get("seed")==v.seed),None)
            execution = recorded or describe_execution(v.route.value,v.backend)
            rel = v.output_path.relative_to(self.out_root)
            q = v.qa
            cls = "bad" if q.critical_errors else ""
            flags = ("FLAGS: " + "; ".join(q.critical_errors)) if q.critical_errors else ""
            from .fidelity_review import review_requirement, review_caption
            fidelity_requirement = review_requirement(v.route.value, q.critical_errors)
            flags = execution['caption'] + '\n' + review_caption(fidelity_requirement) + ("\n" + flags if flags else "")
            if fidelity_requirement['required']:
                cls = "bad"
            review = ('<p class="meta bad">REVIEW REQUIRED: GrabCut fallback may retain source background or cut product edges. '
                      'A high pixel-fidelity score does not validate the mask.</p>'
                      if v.backend == "grabcut" else '')
            href = html.escape(str(rel))
            cards.append(
                f'<div class="shot">{review}<img src="/file/{href}">'
                f'<div class="meta {cls}">{v.route.value} seed {v.seed} - '
                f'{result.request.effective_aspect.value}\n'
                f'auto fidelity {"NOT_EVALUATED (review needed)" if q.score_fidelity is None else str(q.score_fidelity)+"/5"} (structural metric - not a '
                f'human or publishability rating)\n{html.escape(flags)}</div>'
                f'<a class="dl" href="/file/{href}" '
                f'download="{v.route.value}_seed{v.seed}.png">Download PNG</a></div>')
        if result.request.mask_path is not None:
            cards.insert(1, '<p class="meta">SUPPLIED MASK: assisted alpha retained on regenerate. Not automatic segmentation; inspect edges.</p>')
        a = aspect.value if aspect else ""
        route = result.variants[0].route.value if result.variants else "cutout_compose"
        rerun_template = _RERUN
        if result.request.reference_path is not None:
            rerun_template = rerun_template.replace('<h2>Edit / regenerate</h2>',
                '<h2>Edit / regenerate</h2><p class="note">Saved reference/backdrop is retained on regenerate.</p>')
        # Preserve the actual route and alternatives rather than silently
        # switching to first-option defaults on edit/regenerate.
        for value in (r.value for r in Route):
            rerun_template = rerun_template.replace(f'value="{value}">', f'value="{value}"' +
                                                   (' selected' if value == route else '') + '>')
        saved_review = json.loads((job / "source-review.json").read_text())
        for name in ("pose", "material", "framing"):
            pattern = r'(<select name="source_' + name + r'">)(.*?)(</select>)'
            def selected(m):
                return m[1] + m[2].replace('value="' + saved_review[name] + '">',
                    'value="' + saved_review[name] + '" selected>') + m[3]
            rerun_template = re.sub(pattern, selected, rerun_template, flags=re.S)
        if saved_review["require_complete"]:
            rerun_template = rerun_template.replace('name="require_complete" value="yes"', 'name="require_complete" value="yes" checked')
        if intake.get("policy", {}).get("allow_colour_conversion"):
            rerun_template = rerun_template.replace('name="allow_colour_conversion" value="yes"', 'name="allow_colour_conversion" value="yes" checked')
        if intake.get("embedded_alpha_used"):
            rerun_template = rerun_template.replace('name="use_embedded_alpha" value="yes"', 'name="use_embedded_alpha" value="yes" checked')
        n_variants = len(result.variants)
        for value in (2,3,4):
            rerun_template = rerun_template.replace(f'<option>{value}</option>',
                f'<option' + (' selected' if value == n_variants else '') + f'>{value}</option>')
        rerun = rerun_template.format(
            job=job.name,
            options="".join(
                f'<option value="{t.key}"{" selected" if t.key == template_key else ""}>'
                f'{html.escape(t.name)}</option>' for t in all_templates()),
            instruction=html.escape(instruction),
            a11=" selected" if a == "1:1" else "",
            a45=" selected" if a == "4:5" else "",
            a32=" selected" if a == "3:2" else "")
        return _PAGE.format(backend_status=html.escape(getattr(self,"backend_status","Offline stub editor")), options=_template_cards(template_key),
                            result="".join(cards) + rerun).encode()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        if length > 25_000_000:
            self._send(413, b"photo too large (25MB cap)", "text/plain")
            return
        try:
            if path in ("/reviewed-run","/reviewed-confirm"):
                from .reviewed_pipeline import preview_pipeline,finish_pipeline
                origin=self.headers.get("Origin")
                if origin and origin!="http://"+self.headers.get("Host",""):
                    raise ValueError("same-origin local request required")
                lock=getattr(self,"reviewed_lock",None)
                if lock is None or not lock.acquire(blocking=False):raise ValueError("reviewed pipeline busy or not configured")
                try:
                    fields=_parse_multipart(self.rfile.read(length),self.headers.get("Content-Type",""))
                    if path=="/reviewed-run":
                        d=preview_pipeline(self.out_root,fields["job"][1].decode(),json.loads(fields["prompts"][1]),fields["automatic"][1],getattr(self,"reviewed_checkpoint",None))
                    else:
                        d=finish_pipeline(self.out_root,fields["job"][1].decode(),fields["fingerprint"][1].decode(),fields["acceptance"][1].decode())
                    self._send(200,json.dumps(d).encode(),"application/json")
                finally:lock.release()
                return
            if path == "/prompt-prepare":
                from .prompt_editor import prepare_source
                fields = _parse_multipart(self.rfile.read(length),self.headers.get("Content-Type", ""))
                prepared = prepare_source(fields["photo"][1],self.out_root)
                self._send(200,json.dumps(prepared).encode(),"application/json")
                return
            if path == "/run":
                fields = _parse_multipart(self.rfile.read(length),
                                          self.headers.get("Content-Type", ""))
                fname, data = fields["photo"]
                if not data:
                    raise ValueError("empty photo upload")
                ext = Path(fname or "photo.png").suffix or ".png"
                tmp_src = self.out_root / f"_upload{uuid.uuid4().hex[:6]}{ext}"
                self.out_root.mkdir(parents=True, exist_ok=True)
                tmp_src.write_bytes(data)
                template_key = fields.get("template", ("", b"clean_studio"))[1].decode()
                instruction = validate_instruction(fields.get("instruction", ("", b""))[1].decode())
                aspect = _aspect_from(fields.get("aspect", ("", b""))[1].decode())
                variants = int(fields.get("variants", ("", b"2"))[1].decode())
                route = Route(fields.get("route", ("", b"cutout_compose"))[1].decode())
                job, dst, result = self._run_job(tmp_src, template_key,
                                                 instruction, aspect, variants, route,
                                                 ref_upload=fields.get("reference"), mask_upload=fields.get("mask"),
                                                 rgba_upload=fields.get("rgba_result"),
                                                 allow_colour_conversion=fields.get("allow_colour_conversion",("",b""))[1]==b"yes",
                                                 use_embedded_alpha=fields.get("use_embedded_alpha",("",b""))[1]==b"yes",
                                                 source_contract=SourceContract(
                                                     fields.get("source_pose",("",b"unknown"))[1].decode(),
                                                     fields.get("source_material",("",b"unknown"))[1].decode(),
                                                     fields.get("source_framing",("",b"unknown"))[1].decode()),
                                                 require_complete=fields.get("require_complete",("",b""))[1]==b"yes")
                tmp_src.unlink(missing_ok=True)
            elif path == "/rerun":
                form = parse_qs(self.rfile.read(length).decode())
                job_name = form.get("job", [""])[0]
                if not re.fullmatch(r"[a-f0-9]{12}", job_name):
                    raise ValueError("invalid job ID")
                src = next((self.out_root / job_name).glob("source.*"), None)
                if src is None:
                    raise ValueError("job not found or source missing")
                template_key = form.get("template", ["clean_studio"])[0]
                instruction = validate_instruction(form.get("instruction", [""])[0])
                aspect = _aspect_from(form.get("aspect", [""])[0])
                variants = int(form.get("variants", ["2"])[0])
                route = Route(form.get("route", ["cutout_compose"])[0])
                old_reference = next((self.out_root / job_name).glob("reference.*"), None)
                ref_upload = ((old_reference.name, old_reference.read_bytes())
                              if old_reference is not None else None)
                old_mask = self.out_root / job_name / "mask.png"
                mask_upload = ("mask.png", old_mask.read_bytes()) if old_mask.is_file() else None
                old_rgba=self.out_root/job_name/'external-rgba.png'
                rgba_upload=('external-rgba.png',old_rgba.read_bytes())if old_rgba.is_file()else None
                if rgba_upload is not None:mask_upload=None
                prior_path = self.out_root / job_name / "source-review.json"
                prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
                contract = SourceContract(*(form.get("source_"+k,[prior.get(k,"unknown")])[0]
                                           for k in ("pose","material","framing")))
                # New browser form always supplies source_pose; absence means an older API caller.
                complete = (form.get("require_complete",[""])[0]=="yes" if "source_pose" in form
                            else bool(prior.get("require_complete",False)))
                job, dst, result = self._run_job(src, template_key, instruction,
                                                 aspect, variants, route, ref_upload=ref_upload, mask_upload=mask_upload,
                                                 rgba_upload=rgba_upload,source_contract=contract,require_complete=complete,
                                                 allow_colour_conversion=form.get("allow_colour_conversion",[""])[0]=="yes",
                                                 use_embedded_alpha=form.get("use_embedded_alpha",[""])[0]=="yes")
            else:
                self._send(404, b"not found", "text/plain")
                return
        except (ValueError, KeyError, StopIteration, RuntimeError, OSError, zipfile.BadZipFile) as e:
            self._send(400, f"bad request: {html.escape(str(e))}".encode(),
                       "text/plain")
            return
        self._send(200, self._results_page(job, dst, result, template_key,
                                           instruction, aspect))

    def log_message(self, *a):  # keep console clean during tests
        pass


def serve(port: int, out_root: Path, reviewed_checkpoint: Path | None = None, *, editor_kind="stub", mh_model="qwen-edit", live=False) -> ThreadingHTTPServer:
    if editor_kind not in ('stub','magichour') or type(live)is not bool:
        raise ValueError('explicit supported server editor mode required')
    if live and editor_kind!='magichour':raise ValueError('live requires the Magic Hour editor')
    if editor_kind=='magichour':
        from .backends.magichour import MagicHourBackend, MODEL_TABLE
        if mh_model not in MODEL_TABLE:raise ValueError('unknown Magic Hour model')
        def factory(job):return MagicHourBackend(model=mh_model,live=live,cost_log=job/'mh_cost_log.jsonl',receipt_path=job/'mh-projects.jsonl')
        status=('LIVE Magic Hour '+mh_model+'; pure_prompt generates chargeable outputs. Each generate/regenerate can spend credits. Owner approval is required outside this software.' if live else 'MOCK Magic Hour '+mh_model+'; zero-spend offline demonstration, not model quality.')
    else:
        factory=lambda job:StubEditor()
        status='Offline stub editor; zero-spend demonstration, not model quality.'
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    handler = type("BoundHandler", (_Handler,), {"out_root": out_root,"reviewed_checkpoint":reviewed_checkpoint.resolve() if reviewed_checkpoint else None,"reviewed_lock":threading.Lock(),"editor_factory":staticmethod(factory),"backend_status":status,"live_editor":live})
    _render_previews(out_root)  # ready before the first page can request images
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=Path("runs/webui"))
    ap.add_argument("--reviewed-checkpoint",type=Path,help="explicit existing pinned HQ-SAM checkpoint, no download")
    ap.add_argument('--editor',choices=['stub','magichour'],default='stub',help='server-owned editor mode; browser requests cannot change it')
    ap.add_argument('--mh-model',default='qwen-edit')
    ap.add_argument('--live',action='store_true',help='enable chargeable Magic Hour pure_prompt generation; separate owner approval required; never use while paid calls are paused')
    args = ap.parse_args()
    if args.live:print('WARNING: live generation spends credits per request, including regenerate. A flag is not owner authorization.')
    server = serve(args.port, args.out,args.reviewed_checkpoint,editor_kind=args.editor,mh_model=args.mh_model,live=args.live)
    print(f"Renderello UI on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
