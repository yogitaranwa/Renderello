# Constraints

What must not change, and why. If a change breaks one of these, it is not done.

## Preservation

- Fully opaque product pixels are pasted, never resampled twice and never repainted on the deterministic routes. Any intermediate transform voids the contract.
- Reference images steer style and prompt only. Never composite them.
- Template cover images are style previews. Never present them as sample outputs.

## Providers and spend

- No silent provider fallback. An unavailable provider says so; selection is explicit and recorded in the manifest.
- Magic Hour is mock by default. Live calls need the API key, `RENDERELLO_MH_LIVE=1` and a budget that fits the whole batch.
- Reserve budget BEFORE any paid call. A failed or unreachable reserve fails closed.
- Reconcile to the observed charge after every call, up or down. The listed price is a floor.
- Never write the Magic Hour API key to disk, logs, manifests or client responses.
- The Magic Hour editor API has no mask input. Do not claim mask-constrained edits through it.

## Web app

- Single process only: no `--workers`. The job registry, the single-live-Magic-Hour-job rule and the budget reservation lock are per-process by design.
- The job slot claim in `generate` stays before any await, upload or spend.
- No plaintext password storage or comparison. No raw session tokens on disk. No upload contents in logs.
- Static file and run-file serving keep their containment and traversal checks.
- Keep `_SAFE_ERROR_PREFIXES` filtering: client-facing errors stay generic unless the message is known-safe.

## Evidence and review

- Failures stay in manifests, galleries and review denominators.
- Automated QA scores and development-side ratings are never presented as human review or acceptance.
- A human review record states who rated and how that was verified; imported records carry `provenance_verified=false` until checked through the original trusted channel.

## Dependencies and deployment

- No new runtime dependencies without a deliberate pin update in `requirements.txt` (and the matching optional requirement file when applicable).
- No tunnel or proxy serving the Colab notebook as a web backend; Colab's free-runtime terms disallow it.
- No credentials in source, Drive, logs or URLs. Secrets live in the hosting dashboard environment only.
