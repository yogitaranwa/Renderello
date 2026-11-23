#!/usr/bin/env python3
"""End-to-end CPU smoke: synthetic product -> all routes (stub editor) ->
QA -> gallery. Produces real artifacts for eye review. Usage:
  python scripts/smoke_visual.py --out runs/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import make_product                      # noqa: E402
from renderello.backends.base import StubEditor        # noqa: E402
from renderello.config import RunConfig                # noqa: E402
from renderello.gallery import build_gallery           # noqa: E402
from renderello.pipeline import run_pipeline           # noqa: E402
from renderello.schemas import EditRequest, Route      # noqa: E402
from renderello.templates import get_template          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = args.out / "synthetic_product.png"
    make_product().save(src)

    run_dirs = []
    for key in ("clean_studio", "luxury_hero", "social_ad"):
        req = EditRequest(source_path=src, template=get_template(key), seed=42)
        res = run_pipeline(req, [Route.CUTOUT_COMPOSE, Route.PURE_PROMPT,
                                 Route.HYBRID],
                           RunConfig(seed=42, variants=2, out_root=args.out),
                           editor=StubEditor())
        run_dirs.append(res.run_dir)
        for v in res.variants:
            print(f"{key}/{v.route.value}/seed{v.seed}: "
                  f"fidelity={v.qa.score_fidelity} flags={len(v.qa.critical_errors)} "
                  f"-> {v.output_path}")
    g = build_gallery(run_dirs, args.out / "gallery.html")
    print(f"gallery -> {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
