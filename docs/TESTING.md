# Testing and verification checklist

What to run and check before any change counts as done. Actual commands, actual expected outputs.

## Automated suite

```bash
pip install -r requirements.txt
python -m pytest tests/
```

Reference result at the 21 November 2026 checkpoint (`ac6114d`, Python 3.14, pinned requirements): **3,913 passed, 6 skipped, 0 failed**. This is a regression result for the tested code paths, not a photo-quality score or a production load test.

The suite is reproducible as three disjoint invocations over `tests/` when a single run is impractical; record each chunk and do not recount reruns.

## What the suite covers

- Pipeline units: cutout, matting, compositing, presets, prompts, QA metrics.
- Web routes and middleware: body limits, generate gate, job registry, run-file serving and traversal checks (`test_webui.py`, `test_webapp_input_gates.py`, `test_webapp_upload_fuzz.py`, `test_web_rgba_import.py`).
- Auth and revocation: password verify, rate limits, stateless tokens, denylist persistence, corrupt-file and write-failure cases (`test_webapp_auth.py`, `test_webapp_files_and_auth_matrix.py`).
- Budget guards: reserve-before-call, reconcile, batch fit, durable-ledger compare-and-swap.
- Resource behavior: worker pool flatness, stress and memory telemetry (`test_stress.py`, `test_process_memory.py`).
- Security paths and hardening (`test_security_paths.py`, `test_security_hardening.py`).

## Focused checks by change type

| Change | Also run |
|--------|----------|
| Auth, sessions, logout | `pytest tests/test_webapp_auth.py tests/test_webapp_files_and_auth_matrix.py` |
| Generate route or job handling | `pytest tests/test_webapp_input_gates.py tests/test_race_single_job.py` |
| Budget, ledger, Magic Hour config | the budget/ledger test files; confirm mock mode records zero credits |
| Compositing or cutout | the compose/cutout/matting tests; eye-check one real composite per route |
| Preset additions | `pytest tests/test_templates.py tests/test_reviewed_template_coverage.py` |

## Review-sheet tooling (no generation, no network)

```bash
python -m renderello.cli review-export --runs runs/run1 runs/run2 --out review/scores.csv
python -m renderello.cli review-check --sheet review/scores.csv
```

The exporter refuses to overwrite and records each output SHA-256. The checker rejects blank, fractional, out-of-range or conflicting scores, duplicate outputs, changed image bytes and absent rater labels.

## Manual verification before ship

1. Sign in and out on the deployed site; confirm a revoked session stays rejected.
2. Exercise the full authenticated UI in a real browser with the CSP enforcing: login, presets, covers, upload previews, a procedural generation, result images, downloads. Watch the console for violations.
3. Confirm `/api/estimate` shows the cost before Generate on a Magic Hour selection, and that the run refuses when the batch exceeds the budget.
4. Download a result and confirm the fully opaque product region matches the source bit-for-bit.
5. Check the run manifest records provider requested and used, prompt plan, QA verdicts and any failures.

A change that cannot pass its row of this table is not done.
