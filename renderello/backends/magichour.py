"""Magic Hour AI Image Editor backend. Two hard rules baked in:

1. MOCK BY DEFAULT. Without live=True every call is served by an offline mock
   (deterministic gradient echo) and records zero credits. Live calls require
   live=True AND MAGIC_HOUR_API_KEY in the environment. The key is read from
   the environment per call and never written to disk or logs by this repo.
2. READ-ONLY ACCOUNT ENDPOINTS do not generate images or spend credits.
   Permission to access an account must be established outside this code.
   Generation requires live=True; a CLI flag is not owner authorization.

Model/credit table from docs.magichour.ai/api-reference/models
(refreshed 2026-11-20, edit-capable models); free-tier models are 640px-only, NB2.1 is creator-tier. NB2 requests run on 2.1 at 2.1 rates per docs.
Editor API has NO mask input (schema-verified in the plan), so Route 3-style
constrained edits are not available through this backend.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

from PIL import Image

from ..config import MAGICHOUR_API_BASE
from ..schemas import Aspect


# model key -> (credits per image, tier, max long side). Source: plan Part B.
MODEL_TABLE: dict[str, dict] = {
    # Edit-capable models only, from docs.magichour.ai/api-reference/models
    # (fetched 2026-11-14). credits = "From" single-image lowest-res rate;
    # actual charge varies with resolution and image count (docs warn the
    # batch rate is not always count x single rate, e.g. nano-banana-2.1 grid).
    # max_side 640 = free-plan cap; larger resolutions need a paid plan.
    # max_inputs = "Max additional input images (editor)" column.
    "flux-2-klein":    {"credits": 5,   "tier": "free",    "max_side": 640,  "max_inputs": 5},
    "qwen-edit":       {"credits": 10,  "tier": "free",    "max_side": 640,  "max_inputs": 2},
    "krea-2":          {"credits": 10,  "tier": "free",    "max_side": 640,  "max_inputs": 1},
    "qwen-image-2.1":  {"credits": 10,  "tier": "free",    "max_side": 640,  "max_inputs": 2},
    "seedream-v4":     {"credits": 40,  "tier": "creator", "max_side": 4096, "max_inputs": 9},
    "seedream-v4.5":   {"credits": 50,  "tier": "creator", "max_side": 4096, "max_inputs": 9},
    "seedream-v5-flash": {"credits": 30, "tier": "creator", "max_side": 2048, "max_inputs": 9},
    "nano-banana":     {"credits": 50,  "tier": "creator", "max_side": 1024, "max_inputs": 9},
    "nano-banana-2-lite": {"credits": 50, "tier": "creator", "max_side": 1024, "max_inputs": 9},
    "nano-banana-2.1": {"credits": 50,  "tier": "creator", "max_side": 4096, "max_inputs": 9},
    "seedream-v5-pro": {"credits": 75,  "tier": "creator", "max_side": 2048, "max_inputs": 9},
    "nano-banana-2":   {"credits": 50, "tier": "creator", "max_side": 4096, "max_inputs": 9},
    "gpt-image-2": {"credits": 50, "tier": "creator", "max_side": 4096, "max_inputs": 9},
    "gpt-image-2.5-flare": {"credits": 100, "tier": "creator", "max_side": 4096, "max_inputs": 9},
    # Source-verified 13 Nov 2026 receipts (pre-dating the 20 Nov 2026
    # 1000-credit round; 3 receipts = 600 total, earlier baseline scope):
    # 200 credits per image at res=1k, 1:1, single input. Source-code
    # option only - NOT on the live allowlist; a separate pricing/cap
    # analysis is required before any spend on this entry.
    "gpt-image-2.5-flare-1k": {"credits": 200, "tier": "creator", "max_side": 1024, "max_inputs": 1,
                               "api_model": "gpt-image-2.5-flare", "resolution": "1k"},
    "nano-banana-pro": {"credits": 150, "tier": "creator", "max_side": 4096, "max_inputs": 9},
}

_ASPECT_TO_API = {Aspect.SQUARE: "1:1", Aspect.PORTRAIT: "4:5",
                  Aspect.LANDSCAPE: "3:2"}


class MagicHourError(RuntimeError):
    pass


class MagicHourBackend:
    """Editor + read-only account client."""

    def __init__(self, model: str = "qwen-edit", live: bool = False,
                 base_url: str = MAGICHOUR_API_BASE,
                 cost_log: Path | None = None, timeout: float = 120.0,
                 poll_timeout: float = 300.0, poll_interval: float = 3.0,
                 receipt_path: Path | None = None):
        if model not in MODEL_TABLE:
            raise KeyError(f"unknown Magic Hour model '{model}'. "
                           f"known: {', '.join(sorted(MODEL_TABLE))}")
        self.model = model
        self.live = live
        self.base_url = base_url.rstrip("/")
        self.cost_log = cost_log
        self.timeout = timeout
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.receipt_path = receipt_path or Path("runs/mh-projects.jsonl")
        self.last_result = None

    @property
    def name(self):
        return 'magichour:'+self.model+(':live' if self.live else ':mock')

    def attempt_evidence(self):
        """Sanitized per-call facts, not provider/account authentication or price.

        Only a current terminal detail supplies settled recorded charge. Submission
        and pending charges are separate observations, never a zero-cost fallback.
        """
        import math
        def charge(row):
            value = row.get('credits_charged') if isinstance(row,dict) else None
            return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None
        state = self.last_result if isinstance(self.last_result,dict) else {}
        detail = state.get('final') or state.get('latest') or {}
        status = detail.get('status')
        terminal = status in ('complete','error','canceled')
        project = state.get('id')
        if not isinstance(project,str) or not project or len(project)>240:project=None
        return dict(schema='provider-attempt.v1',provider='magichour',model=self.model,
            attempt_id=getattr(self,'_attempt_id',None),mock=not self.live,
            stage=getattr(self,'_attempt_stage','not_started'),project_id=project,
            provider_status=status if status in ('complete','error','canceled','draft','queued','rendering') else None,
            terminal_detail_observed=terminal,
            credits_charged=charge(detail) if terminal else None,
            submission_credits_observed=charge(state.get('submission')),
            latest_credits_observed=charge(detail),
            charge_basis='terminal_detail' if terminal else 'UNKNOWN_NOT_SETTLED',
            receipt_path=str(self.receipt_path),cost_log_path=None if self.cost_log is None else str(self.cost_log),
            account_credit_system='NOT_BOUND_REQUIRES_CALLER_ACCOUNT_SCOPE',
            semantics='recorded_backend_response_not_provider_authentication; no_price_or_profit_inferred')

    # ---- read-only account ----
    def balance(self) -> dict:
        """GET /v1/account - read-only, no generation. Account-access authority
        must be checked separately. Returns the provider JSON."""
        import os
        import requests
        key = os.environ.get("MAGIC_HOUR_API_KEY")
        if not key:
            raise MagicHourError("MAGIC_HOUR_API_KEY not set; balance is read-only "
                                 "but still needs the key in the environment")
        r = requests.get(f"{self.base_url}/v1/account",
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=self.timeout)
        if r.status_code != 200:
            raise MagicHourError(f"balance check failed: HTTP {r.status_code}")
        return r.json()

    # ---- generation (gated) ----
    def credits_per_image(self) -> int:
        return MODEL_TABLE[self.model]["credits"]

    def edit(self, source: Image.Image, prompt: str, negative_prompt: str,
             seed: int, aspect: Aspect, reference: Image.Image | None = None,
             extra_images: list[Image.Image] | None = None) -> Image.Image:
        """One editor generation. Inputs: product photo first, then any extra
        product photos and/or style references. Multi-image models take up to
        max_inputs additional images; overflow is noted, never silently dropped.
        """
        import uuid
        self.last_result = None
        self._attempt_id = uuid.uuid4().hex
        self._attempt_stage = 'before_submission'
        extras = list(extra_images or [])
        if reference is not None:
            extras = [reference] + extras
        limit = MODEL_TABLE[self.model].get("max_inputs", 0)
        self.dropped_inputs = []
        if len(extras) > limit:
            self.dropped_inputs = extras[limit:]
            extras = extras[:limit]
        try:
            image = self._edit_once(source, prompt, negative_prompt, seed, aspect,
                                    extra_images=extras)
            self._attempt_stage = 'output_decoded'
            return image
        except Exception:
            # A transport error after POST can still be billed. Never retry here.
            raise

    def _edit_once(self, source, prompt, negative_prompt, seed, aspect, reference=None,
                   extra_images=None):
        if not self.live:
            self.last_result = {'mock': True, 'final': {'status': 'complete', 'credits_charged': 0}}
            return self._mock_edit(source, seed, aspect)
        import os
        import requests
        key = os.environ.get("MAGIC_HOUR_API_KEY")
        if not key:
            raise MagicHourError("live edit requested but MAGIC_HOUR_API_KEY is unset")
        extras = list(extra_images or [])
        if reference is not None:
            extras = [reference] + extras
        images = [source] + extras
        headers = {"Authorization": f"Bearer {key}"}
        paths = []
        for image in images:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, "PNG")
            up = requests.post(f"{self.base_url}/v1/files/upload-urls",
                               headers=headers,
                               json={"items": [{"type": "image", "extension": "png"}]},
                               timeout=self.timeout)
            up.raise_for_status()
            item = up.json()["items"][0]
            requests.put(item["upload_url"], data=buf.getvalue(),
                         headers={"Content-Type": "image/png"},
                         timeout=self.timeout).raise_for_status()
            paths.append(item["file_path"])
        # Lowest supported resolution fixed explicitly: auto is tier-dependent
        # and can silently cost more. API has no seed or negative_prompt fields.
        entry = MODEL_TABLE[self.model]
        payload = {"model": entry.get("api_model", self.model), "image_count": 1,
                   "resolution": entry.get("resolution", "1k" if self.model == "nano-banana-pro" else "640px"),
                   "aspect_ratio": _ASPECT_TO_API[aspect],
                   "style": {"prompt": prompt},
                   "assets": {"image_file_paths": paths}}
        started = time.monotonic()
        # Never automatically retry a generation POST, including transport errors:
        # the server might already have accepted and charged it.
        self._attempt_stage = 'submission_started'
        res = requests.post(f"{self.base_url}/v1/ai-image-editor",
                            headers=headers, json=payload, timeout=self.timeout)
        if res.status_code == 402:
            raise MagicHourError("insufficient credits (402): no automatic retry")
        if res.status_code in (401, 403):
            raise MagicHourError(f"auth/plan gate (HTTP {res.status_code}): no retry")
        res.raise_for_status()
        accepted = res.json()
        project_id = accepted.get("id")
        if not project_id:
            raise MagicHourError("submission response lacks project ID; do not resubmit blindly")
        self.last_result = {"id": project_id, "submission": accepted}
        self._attempt_stage = 'project_accepted'
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with self.receipt_path.open("a") as f:
            f.write(json.dumps({"project_id": project_id, "model": self.model,
                                "credits_charged": accepted.get("credits_charged"),
                                "resolution": payload["resolution"]}) + "\n")
        # Persist receipt before polling/download so recovery never needs a second
        # generation. Creation charge and final/refunded charge remain distinct.
        self.last_result = {"id": project_id, "submission": accepted}
        final = self.wait_project(project_id)
        self.last_result = {"id": project_id, "submission": accepted, "final": final}
        self._log_cost(seed, aspect, time.monotonic() - started, final)
        return self._download(final)

    def wait_project(self, project_id: str) -> dict:
        """Read-only bounded status polling. Timeout preserves ID; no recreation."""
        import os
        import requests
        from urllib.parse import quote
        key = os.environ.get("MAGIC_HOUR_API_KEY")
        if not key:
            raise MagicHourError("MAGIC_HOUR_API_KEY not set")
        deadline = time.monotonic() + self.poll_timeout
        while True:
            res = requests.get(f"{self.base_url}/v1/image-projects/{quote(project_id, safe='')}",
                               headers={"Authorization": f"Bearer {key}"},
                               timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            if isinstance(self.last_result, dict) and self.last_result.get('id') == project_id:
                self.last_result['latest'] = data
            state = data.get("status")
            if state == "complete":
                return data
            if state in ("error", "canceled", "draft"):
                raise MagicHourError(f"project {project_id} status={state}; recover this ID, do not resubmit")
            if state not in ("queued", "rendering"):
                raise MagicHourError(f"project {project_id} returned unknown status {state!r}")
            if time.monotonic() >= deadline:
                raise MagicHourError(f"project {project_id} still {state}; polling timed out, resume this ID")
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))

    def _download(self, project: dict) -> Image.Image:
        import requests
        downloads = project.get("downloads") or []
        if not downloads:
            raise MagicHourError("complete project has no downloads; recover project, do not resubmit")
        res = requests.get(downloads[0]["url"], timeout=self.timeout)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")

    def resume(self, project_id: str) -> Image.Image:
        """Recover existing project by ID, read-only, never a new generation."""
        final = self.wait_project(project_id)
        self.last_result = {"id": project_id, "final": final}
        return self._download(final)

    def _mock_edit(self, source: Image.Image, seed: int, aspect: Aspect) -> Image.Image:
        """Offline deterministic echo: resize + seed-dependent tint. Exercises
        orchestration and cost accounting with zero network and zero credits."""
        import numpy as np
        arr = np.array(source.convert("RGB").resize(aspect.size,
                                                    Image.LANCZOS)).astype(np.float32)
        arr[..., 2] = np.clip(arr[..., 2] + (seed % 5) * 3.0, 0, 255)
        self._log_cost(seed, aspect, 0.0, {"mock": True, "credits_charged": 0})
        return Image.fromarray(arr.astype(np.uint8))

    def _log_cost(self, seed: int, aspect: Aspect, latency_s: float,
                  response: dict) -> None:
        if self.cost_log is None:
            return
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.model, "seed": seed, "aspect": aspect.value,
            "credits_listed": self.credits_per_image(),
            "credits_charged": response.get("credits_charged", None if self.live else 0),
            "mock": not self.live, "latency_s": round(latency_s, 3),
        }
        with self.cost_log.open("a") as f:
            f.write(json.dumps(row) + "\n")
