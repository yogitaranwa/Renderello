# Handover

Where things stand, 21 November 2026. Read this first; update it at the end of every session.

## Done

- Web studio deployed at https://renderello-studio.onrender.com: email/password login, template picker with per-template options, cost estimate before Generate, guarded Magic Hour engine, one verified live 30-credit scene-compose run.
- Self-contained Colab notebook at the handover root, clean (no saved outputs or execution counts); an executed copy with retained outputs from the verified Magic Hour run is in `docs/`.
- Twelve researched shoot templates, deterministic compositing with bit-exact opaque-pixel preservation, automated fidelity QA, benchmark and gallery tooling.
- Test suite green at the checkpoint: 3,913 passed, 6 skipped (Python 3.14, pinned requirements).
- Existing-results gallery in `outputs/` with a first-look selection, the 480-case procedural diagnostic set, historical Magic Hour comparisons, specialist examples and failures kept visible.

## In progress

- Nothing actively running. The tree is at a deliberate stop point for review and human evaluation.

## Broken or open

- Long local stress runs show memory growth (about 160 MB to 700 MB over roughly 2,000 generations); cause not found. The one-job cap and bounded pool bound the blast radius.
- A clean fresh-runtime full Run all of the notebook has not been verified end to end; expect kernel restarts on a fresh runtime.
- Free-host reliability over long sessions is unproven; generated results are ephemeral (24 h janitor, redeploy wipes).
- Cross-instance session revocation after instance replacement is bounded by the 12 h token expiry, not solved.
- The free-GPU photographic path in the notebook is experimental (grayscale/illustrated backgrounds, OOM and dependency conflicts seen).

## Not established (do not claim)

- The brief's 80 percent no-critical-alteration and 70 percent publication-usefulness gates.
- A completed matched comparison against two leading specialist tools reviewed by a human.
- Polished, genuinely distinct template quality for eight or more shoots.
- A verified tiny-text fidelity case.
- Human review of the current outputs; existing scores are development-side and await it.

## Avoid

- Paid Magic Hour calls without a fresh, explicit budget approval. The live flag plus budget guards exist so an accidental run cannot spend, but do not test that casually.
- Running more than one generation job on the free instance (OOM risk, measured).
- Trusting forwarded IP headers for rate limiting; the global breaker is deliberate (see DECISIONS.md).
