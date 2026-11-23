"""Shared data structures. Everything crossing module boundaries is one of these,
so the benchmark, QA and gallery all speak the same record shapes."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class Route(str, Enum):
    PURE_PROMPT = "pure_prompt"          # Route 1: whole-image generative edit
    CUTOUT_COMPOSE = "cutout_compose"    # Route 2: deterministic cutout + composite
    HYBRID = "hybrid"                    # Route 3: composite + constrained harmonize
    SCENE_COMPOSE = "scene_compose"      # Route 2b: relit/generated scene + source-exact product paste (needs scene_fn, GPU in practice)


class Aspect(str, Enum):
    SQUARE = "1:1"
    PORTRAIT = "4:5"
    LANDSCAPE = "3:2"

    @property
    def size(self) -> tuple[int, int]:
        # Working canvases are multiples of 32 with the EXACT labeled ratio.
        # Keep the longest side <=1120; model adapters handle their own padding.
        return {Aspect.SQUARE: (1024, 1024),
                Aspect.PORTRAIT: (896, 1120),
                Aspect.LANDSCAPE: (1056, 704)}[self]


@dataclass(frozen=True)
class TemplateSpec:
    """One reusable shoot template. Prompt fragments stay composable so custom
    instructions append without rewriting the fidelity guards."""
    key: str
    name: str
    category: str
    scene: str            # environment / surface / props
    lighting: str         # light direction, quality, color temperature
    camera: str           # lens / angle / framing language
    mood: str
    aspect: Aspect
    background_hint: str  # guides procedural background + cutout-route grading
    shadow: str           # shadow behavior the composite must honor


@dataclass(frozen=True)
class EditRequest:
    """One generation: source + template + optional owner instruction + seed."""
    source_path: Path
    template: TemplateSpec
    custom_instruction: str = ""
    seed: int = 0
    aspect: Aspect | None = None        # override; default = template aspect
    reference_path: Path | None = None  # optional inspiration/backdrop photo (PDF section 4, optional)
    mask_path: Path | None = None       # optional explicit alpha, source-sized L-mode PNG

    @property
    def effective_aspect(self) -> Aspect:
        return self.aspect or self.template.aspect


@dataclass
class CutoutResult:
    rgba_path: Path                     # product with alpha matte
    matte_path: Path                    # 8-bit alpha matte (255 = product)
    bbox: tuple[int, int, int, int]     # x, y, w, h of the product in source coords
    backend: str                        # birefnet / rmbg2 / grabcut
    coverage: float                     # matte area fraction, sanity signal


@dataclass
class QAReport:
    """Fidelity of one output vs its source. Product-region metrics are computed
    only inside the (aligned) product matte; 'pixel-exactness' claims are valid
    only for untransformed interior pixels (Route 2)."""
    source: str
    output: str
    route: str
    template: str
    seed: int
    ssim_product: float | None = None
    l1_product: float | None = None
    color_drift: float | None = None    # Bhattacharyya distance, product region
    edge_retained: float | None = None  # fraction of source edges present in output
    critical_errors: list[str] = field(default_factory=list)
    score_fidelity: int | None = None   # 1-5 rubric (plan Part A); None = unscored
    notes: str = ""

    def to_json(self) -> str:
        # JSON has no NaN/Infinity. Missing metric evidence is null, not zero
        # or an invented pass, and the original report remains unchanged.
        data = asdict(self)
        for field in ('ssim_product', 'l1_product', 'color_drift', 'edge_retained'):
            value = data[field]
            if value is not None and not math.isfinite(value):
                data[field] = None
        return json.dumps(data, indent=2, allow_nan=False)


@dataclass
class VariantResult:
    """One produced alternative with everything needed to audit it."""
    route: Route
    seed: int
    output_path: Path
    qa: QAReport
    intermediates: dict[str, Path] = field(default_factory=dict)
    backend: str = ""
    credits_estimated: int = 0          # Magic Hour routes only; 0 for local routes
    seconds: float = 0.0                # per-variant wall time (latency metric)
    success: bool = True                 # executor/QA completed, not photo acceptance
    provider_evidence: dict | None = None # actual recorded per-attempt facts; unknown is not zero


@dataclass
class RunResult:
    request: EditRequest
    variants: list[VariantResult]
    run_dir: Path
    manifest_path: Path

    def best_by_qa(self) -> VariantResult | None:
        scored = [v for v in self.variants if v.success and v.output_path.is_file() and v.qa.score_fidelity is not None]
        return max(scored, key=lambda v: v.qa.score_fidelity) if scored else None
