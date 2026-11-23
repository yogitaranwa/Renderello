"""Blank review-sheet export and strict scored-sheet import. No automatic ratings.
Rater labels record a claim; they do not authenticate a human or grant authority.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from .artifact_paths import run_artifact
from pathlib import Path

DIMENSIONS = ("product_label_fidelity", "photorealism", "shadow_contact_realism",
              "material_edges", "composition", "template_distinctness",
              "publish_ready_usefulness")
IDENTITY = ("source", "output", "output_sha256", "template", "route", "seed")
FIELDS = (*IDENTITY, *DIMENSIONS, "no_critical", "critical_errors", "rater", "notes")
RATERS = {"owner", "independent_human", "agent_eye"}


def export_sheet(run_dirs: list[Path], output: Path, portable: bool = False) -> Path:
    """Export existing readable outputs once. Never overwrite a scored sheet."""
    rows = []
    seen = set()
    for run in run_dirs:
        manifest = json.loads((run / "manifest.json").read_text())
        for variant in manifest.get("variants", []):
            p = run_artifact(run, variant.get("output", ""))
            if p is None:
                continue  # No image to rate; generation failures stay in manifests.
            p = p.resolve()
            if str(p) in seen:
                raise ValueError(f"duplicate output: {p}")
            seen.add(str(p))
            rows.append({"source": manifest.get("source", ""), "output": (Path(os.path.relpath(p, output.parent.resolve())).as_posix() if portable else str(p)),
                         "output_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "template": variant.get("qa", {}).get("template", ""),
                         "route": variant.get("route", ""), "seed": variant.get("seed", "")})
    if not rows:
        raise ValueError("no existing outputs to review")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)  # All rating fields intentionally blank.
    return output


def load_scores(sheet: Path) -> dict:
    """Validate complete ratings and exact image bytes; never infer blank values.

    Returns records compatible with benchmark._summarize(human=...), plus all
    seven dimensions. Caller still owns trusted-channel rater provenance.
    """
    scores = {}
    with sheet.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("review sheet lacks required fields")
        for row in reader:
            key = row["output"]
            if key in scores:
                raise ValueError(f"duplicate output: {key}")
            p = Path(key)
            if not p.is_absolute():
                p = sheet.parent / p
            p = p.resolve()
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["output_sha256"]:
                raise ValueError(f"output missing or bytes changed: {key}")
            dims = {}
            for name in DIMENSIONS:
                if row[name] not in {"1", "2", "3", "4", "5"}:
                    raise ValueError(f"{key}: {name} must be a complete integer rating 1-5")
                dims[name] = int(row[name])
            if row["no_critical"] not in {"true", "false"}:
                raise ValueError(f"{key}: no_critical must be true or false")
            if row["rater"] not in RATERS:
                raise ValueError(f"{key}: explicit rater provenance required")
            errors = row["critical_errors"].strip()
            no_critical = row["no_critical"] == "true"
            if no_critical == bool(errors):
                raise ValueError(f"{key}: critical error text conflicts with no_critical")
            canonical_key = str(p)
            if canonical_key in scores:
                raise ValueError(f"duplicate output: {key}")
            scores[canonical_key] = {"no_critical": no_critical,
                           "publishable": dims["publish_ready_usefulness"],
                           "rater": row["rater"], "dimensions": dims,
                           "critical_errors": errors, "notes": row["notes"],
                           "provenance_verified": False}
    if not scores:
        raise ValueError("review sheet contains no ratings")
    return scores
