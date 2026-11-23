# Architecture

The system map for Renderello: modules, layers and how data moves between them. Implementation detail lives in `TECHNICAL_DOCUMENTATION.md`; this is the shape of the thing.

## Layers and dependency direction

```
┌─────────────────────────────────────────────────────────────────┐
│ Entry surfaces                                                   │
│   webapp/main.py (FastAPI)   Renderello_Studio.ipynb   cli.py    │
└───────────────┬───────────────────────┬─────────────────────────┘
                v                       v
┌─────────────────────────────────────────────────────────────────┐
│ Orchestration                                                    │
│   studio.py  run_studio: one generate call, website + notebook   │
│   pipeline.py  run_pipeline: route x variant research harness    │
└───────┬───────────────┬───────────────┬─────────────────────────┘
        v               v               v
┌──────────────┐ ┌──────────────┐ ┌────────────────────────────────┐
│ Domain       │ │ Providers    │ │ Imaging core                   │
│ presets.py   │ │ scene_       │ │ image_intake -> cutout ->      │
│ templates.py │ │ providers.py │ │ matting -> compose             │
│ prompts.py   │ │ backends/    │ │ (paste, never repaint)         │
│ longprompt.py│ │  magichour   │ └──────────────┬─────────────────┘
└──────────────┘ │  qwen_local  │                v
                 │  klein_local │        qa.py (masked metrics)
                 └──────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Platform guards        webapp/auth.py, mh_ledger.py,             │
│                        process_memory.py, resources.py           │
│ Evidence and review    benchmark.py, gallery.py, review.py,      │
│                        brief_acceptance.py, brief_economics.py,  │
│                        human review pack modules                 │
└─────────────────────────────────────────────────────────────────┘
```

Rules of the map:

- Entry surfaces import the package; the package never imports the webapp. There is no circular path back up.
- `schemas.py` and `config.py` are the bottom layer: shared record shapes (`EditRequest`, `CutoutResult`, `QAReport`, `VariantResult`, `RunResult`) and model IDs. Every layer above speaks these shapes.
- Providers generate scenes or whole-image edits only. Product pixels reach the canvas through `compose`, never through a generative model, on the deterministic routes.
- `webapp/auth.py` is imported only by `webapp/main.py`; nothing in the package knows sessions exist.
- `mh_ledger.py` is the only module that talks to the ledger store; spend flows studio -> ledger, never the reverse.

## How the layers communicate

- Web to orchestration: direct function call (`run_studio`) on a bounded worker pool; results come back through run directories and manifests on disk, polled via `GET /api/jobs/{job_id}`.
- Orchestration to providers: the `SceneProvider` interface (`available`, `generate_scene`, cost and quality notes) or the editor backend interface; selection is explicit per run, recorded in the manifest.
- Orchestration to disk: each run is a directory with the source copies, intermediates, variants, contact sheet and `manifest.json`. The manifest is the audit record: provider requested and used, prompt plan, QA verdicts, warnings, failures.
- No message queues, no background jobs beyond the in-process worker pool and the 24 h janitor thread, no external calls except the Magic Hour API (paid, guarded) and the ledger store (compare-and-swap writes only).

## What lives where

| Question | Answer |
|----------|--------|
| HTTP routes, middleware, job registry | `webapp/main.py` |
| Credentials, sessions, rate limits, logout denylist | `webapp/auth.py` |
| The flow the website and notebook share | `renderello/studio.py` |
| Template catalog and prompt assembly | `renderello/presets.py`, `prompts.py`, `longprompt.py` |
| Cutout, matting, compositing | `renderello/cutout.py`, `matting.py`, `compose.py` |
| Scene/editor backends | `renderello/scene_providers.py`, `renderello/backends/` |
| Automated fidelity metrics | `renderello/qa.py` |
| Spend budget and durable ledger | `renderello/studio.py` (budget), `renderello/mh_ledger.py` (durable store) |
| Benchmark and evidence tooling | `renderello/benchmark.py`, `gallery.py`, `review.py`, `cli.py` |
| Human review tooling | `renderello/blind_review.py`, `blind_review_return.py` (module names predate current terminology), see `HUMAN_REVIEW.md` and `HUMAN_REVIEW_PACKS.md` |
