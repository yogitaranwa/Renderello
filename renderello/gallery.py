"""Private gallery generator. Static HTML, no external assets, relative paths -
open the folder locally or share the Drive folder privately. Per the plan, the
gallery stays private (no public Space/publish)."""
from __future__ import annotations

import html
import hashlib
import json
import os

from .artifact_paths import run_artifact
from .fidelity_review import review_requirement, review_caption
from pathlib import Path

_CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222;
overflow-x:hidden}
h1{font-size:20px} h2{font-size:16px;margin-top:32px}
.note{font-size:12px;color:#777;max-width:900px}
.pair{display:inline-block;vertical-align:top;margin:8px;background:#fff;
border:1px solid #ddd;border-radius:8px;padding:8px;width:340px;max-width:94%}
.pair img{width:100%;border-radius:4px}
.meta{overflow-wrap:anywhere;font-size:12px;color:#555;margin-top:6px;white-space:pre-wrap}
.bad{color:#a00}.good{color:#070}
"""

_DISCLAIMER = ("All scores are AUTOMATED STRUCTURAL metrics (pixel-level fidelity "
               "checks), not human ratings and not publish-ready judgments. "
               "Source classification, attribution and rights status are shown per source below. "
               "Unknown inputs are not assumed synthetic or licensed. This gallery alone "
               "does not establish the full benchmark gate.")


def _source_caption(m: dict) -> str:
    intake = m.get('image_intake') or {}
    basis = ('\nIMAGE INTAKE: ' + ', '.join(intake.get('transformations') or ['decoded RGB assumed sRGB']) +
             '. Product pixel comparisons use normalized RGB, not original encoded colour values. Colour accuracy is not verified.') if intake else ''
    p = m.get('source_provenance')
    if not p:
        return ('SOURCE: UNKNOWN / unclassified legacy run. No content-bound source record. '
                'Do not assume synthetic, licensed or publishable; rerun to record provenance.' + basis)
    return '\n'.join(['SOURCE: ' + p.get('classification', 'UNKNOWN / unclassified'),
                      'Content match: ' + p.get('match_basis', 'unrecorded'),
                      'SHA256: ' + p.get('sha256', 'unrecorded'),
                      'Attribution: ' + p.get('attribution', 'unrecorded'),
                      'Record: ' + p.get('record_origin', 'unrecorded'),
                      'Rights: ' + p.get('rights_status', 'unrecorded')]) + basis


def build_gallery(run_dirs: list[Path], out_html: Path, title: str = "Renderello gallery (private)",
                  source_paths: dict[str, Path] | None = None) -> Path:
    sections: list[str] = []
    for rd in run_dirs:
        manifest = rd / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        tpl = m["template"]["name"]
        rows: list[str] = [f"<h2>{html.escape(tpl)} - {html.escape(Path(m['source']).name)}</h2>"]
        source = Path(m["source"])
        digest = m.get("source_provenance", {}).get("sha256")
        if source_paths and digest in source_paths:
            source = Path(source_paths[digest])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("source override missing or hash mismatch: " + digest)
        if source.is_file():
            rows.append('<div class="pair"><img src="{}" alt="source"><div class="meta">{}</div></div>'.format(
                html.escape(_rel(str(source), out_html.parent)), html.escape(_source_caption(m))))
        else:
            rows.append('<div class="pair"><div class="meta bad">Source image not included in this archive. '
                        'No before/after comparison is claimed.</div><div class="meta">' +
                        html.escape(_source_caption(m)) + '</div></div>')
        for v in m["variants"]:
            q = v["qa"]
            output = run_artifact(rd, v.get("output", ""))
            output_exists = output is not None
            from .completion import saved_variant_failed
            failed = saved_variant_failed(v)
            if not output_exists or failed:
                failures = q.get("critical_errors") or [m.get("cutout_error") or "Output file is missing"]
                text = (("FAILED / PARTIAL ARTIFACT - NOT A COMPLETED OUTPUT\n" if output_exists else "FAILED / NOT GENERATED\n") +
                        f"route {v.get('route', 'unknown')} | seed {v.get('seed', 'unknown')}\n" +
                        ("Artifact retained for debugging. No quality rating.\n" if output_exists else "No image output and no quality rating.\n") + "; ".join(str(e) for e in failures))
                rows.append(f'<div class="pair"><div class="meta bad">{html.escape(text)}</div></div>')
                continue
            score = q.get("score_fidelity")
            errs = q.get("critical_errors") or []
            cls = "bad" if errs or v.get("backend") == "grabcut" else ""
            meta = (f"route {v['route']} | seed {v['seed']} | backend {v.get('backend','')}\n"
                    f"auto fidelity {'NOT_EVALUATED (review needed)' if score is None else str(score)+'/5'} (structural metric, not a human rating) | "
                    f"ssim {_fmt(q.get('ssim_product'))} | "
                    f"drift {_fmt(q.get('color_drift'))} | edges {_fmt(q.get('edge_retained'))}")
            if v.get("backend") == "grabcut":
                meta += "\nREVIEW REQUIRED: GrabCut fallback may retain source background or cut product edges. A high pixel-fidelity score does not validate the mask."
            if m.get("mask_provenance") == "embedded_source_alpha":
                meta += "\nEMBEDDED SOURCE ALPHA: explicitly retained after intake; no segmentation model; mask and material semantics unverified."
            elif v.get("backend") == "supplied_mask":
                meta += ("\nASSISTED SEEDED GRABCUT MASK: explicit correction regions/strokes, not automatic success or original GPU alpha."
                         if m.get("mask_provenance") == "estimated_seeded_grabcut_assisted" else
                         f"\nESTIMATED {m.get('mask_provenance','').replace('estimated_','').upper()} MASK: not original GPU alpha; inspect missing objects, halos and transparency."
                         if (m.get("mask_provenance") or "").startswith("estimated_") else
                         "\nSUPPLIED MASK: explicit assisted alpha, not automatic segmentation; inspect edges.")
            from .execution_kind import describe_execution
            execution = v.get('execution_kind') or describe_execution(v.get('route'),v.get('backend',''))
            meta += '\n' + execution['caption']
            requirement = review_requirement(v.get('route'), errs)
            meta += "\n" + review_caption(requirement)
            if requirement['required']:
                cls = "bad"
            if errs:
                meta += "\nFLAGS: " + "; ".join(errs)
            rows.append(
                f'<div class="pair"><img src="{html.escape(_rel(str(output), out_html.parent))}">'
                f'<div class="meta {cls}">{html.escape(meta)}</div></div>')
        sections.append("\n".join(rows))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
           f"<body><h1>{html.escape(title)}</h1>"
           f"<p class='note'>{html.escape(_DISCLAIMER)}</p>"
           + "\n".join(sections) + "</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc)
    return out_html


def _rel(p: str, base: Path) -> str:
    # A basename fallback silently broke galleries outside the run directory.
    # The gallery and referenced folders must travel together when sharing.
    return Path(os.path.relpath(Path(p).resolve(), base.resolve())).as_posix()


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
