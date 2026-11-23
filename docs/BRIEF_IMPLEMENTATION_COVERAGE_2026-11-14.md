# Brief implementation coverage and remaining work
14 November 2026, 09:25 IST. This is an implementation audit, not photographic acceptance.

## Working product entry points

The existing dashboard has source upload, ten configured template tiles, optional 16,000-character scene instructions, 2-4 alternatives, square/portrait/landscape selection, optional reference backdrop, comparison/download and edit/regenerate. Tiles remain synthetic demonstrations, not eight accepted real covers.

A genuine integration gap was found: dashboard generation previously hard-coded StubEditor while the command-line pipeline supported Magic Hour. It now supports a server-owned editor/model setting:

```
python -m renderello.webui --editor magichour --mh-model nano-banana-2.1 --out runs/ui-mock
```

This remains MOCK and makes no provider request. Default without arguments is still the stub. A separate `--live` startup flag exists for future authorized testing only. Do not use it while paid calls are paused. The server binds to localhost. Browser form parameters cannot select the backend/model/live mode. Generate and regenerate can both spend credits on a live-configured server; startup flags never establish owner permission. No real backend was invoked during this change. A configured live server is not a production release, user-authenticated credit dashboard or spend-budget mechanism.

Provider model settings are server-wide for a bounded demo, not per-template accepted choices. Per-job cost logs and project receipt paths are retained. Magic Hour now exposes a recorded model/mock-or-live identifier, so saved pipeline variants do not degrade to the generic backend label `editor`. Offline echo outputs are clearly classified as test doubles, never quality evidence. Recorded live configuration is not proof of completion or actual charge; provider receipts remain necessary. Live hybrid is refused because this API cannot provide the required mask constraint. Pure prompting remains a whole-product edit with fidelity-review requirements.

Existing CLI `run --editor magichour` and the comparison runner remain available. No accepted best/value model is preselected.

## New tools are reachable from the main CLI

```
python -m renderello.cli brief-economics --input evidence.json --output new-report.json
python -m renderello.cli blind-review-pack --manifest pack.json --out new-pack
python -m renderello.cli blind-review-return --pack new-pack --returned-csv separate-return.csv --provenance claimed_independent --review-reference session-reference --output private-return.json
python -m renderello.cli native-region-matte --source native-rgb.png --alpha prior-alpha.png --spec reviewed-roi.json --out new-result
```

The source/destination arguments and underlying strict contracts are described in the economics, blind-review and native-region docs. Each command has `--help`. Outputs are exclusive; older artifacts are not replaced. A blind-pack manifest declares source, seed, candidates and optional regions; relative input paths resolve against that manifest. Candidate entries declare method, path or explicit failure. Packs do not make native aspect-mismatched specialist outputs registered or fine-print comparable.

Native-region spec requires encoded source_file_sha256, alpha_file_sha256, roi [x0,y0,x1,y1], reviewed true, review_note, material opaque/opaque_fibres, method cf/lkm/guided, edge_colors boolean and feather integer. Optional --trimap requires trimap_file_sha256 in the spec and an ROI-sized plain grayscale trimap. Source coordinates must already be orientation-normalized. No silent ICC conversion, resizing or semantic selection occurs.

Native-region file output includes a plain normalized-source.png, alpha.png, candidate.png, retained original source bytes, request, hash report and native white/dark/magenta ROI checks. To continue through the existing dashboard, upload normalized-source.png and alpha.png in its explicit mask input, then make the source pose/material/framing declarations. This carries the refined ALPHA, not estimated boundary foreground colours. Existing RGBA import has a 1MP cap, so it is not the larger-garment route. candidate.png with edge_colors true remains a separate research candidate, never silently accepted as source-exact RGB. The ROI function has size/band checks, not a wall-clock/RSS worker guarantee.

## What is still manual or unimplemented

- Economics and returned blind-review workflows are researcher CLI/Python tools, not dashboard upload/result panels. No requirement for a separate evaluation-admin UI appears in the brief. This choice keeps the bounded product demo simple, but must not be described as a fully integrated dashboard evaluation experience.
- Run manifests/provider receipts are not automatically converted into a preregistered economics plan, pricing proof or independent review. Assemble the typed plan/attempt records explicitly, retaining receipt and review references. `bind_ratings` validates the declared token-to-output mapping against source/method/native output/first-try constraints. It does not authenticate a reviewer. Automatic adapters would need an explicit common attempt schema and recovery policy to avoid inventing charges or dropping failed jobs.
- Native-region selection/trimap preparation is not a dashboard paint tool and is not automatic semantic segmentation. It is a local reviewed preprocessing step. The brief asks to evaluate masks/compositing, not build a Photoshop-like editor.
- Reference images use the existing backend API handling; supported ratios, current model schema, charge/recovery behavior and whole live upload-to-download/regenerate must be verified in a future authorized live run. Mock route checks cannot establish them.
- No production auth/billing/Shopify/bulk rollout was added; these are explicitly outside the prototype scope or require later review.

## Evidence still required, not closable by more software tests

Twenty qualified varied real sources with evaluated outputs; eight polished distinct reusable covers; matched first-try comparisons with two leading specialists where trials permit; complete seven-dimension/critical-error judgments and honest provenance; visually competitive acceptance; current verified credit valuation, retry-inclusive costs, usable-output unit economics and credible sale/other-cost assumptions. Existing reports deliberately leave these unknown. Productization remains NO-GO.

The local garment ROI replay is a small supervised boundary candidate, not another accepted source/template or broad fidelity solution. Most visible smoothing was already in the earlier CF alpha crop. Fringe/dark-boundary ambiguity and other full-image defects remain.

## Validation and limits

The first combined entrypoint+dashboard run ended with 21 progress dots then two failures and no usable exception detail when storage reached zero. Preserve it as inconclusive, not a full pass or proven disk-only failure. After removing only already cloud-download/SHA/CRC-verified ZIP packaging under the standing grant, the exact last two tests passed with preview generation replaced by tiny tiles to limit storage. Their real pipeline paths were retained. This does not validate original full-size previews again.

Separate new tests check CLI round trips, source/hash/mode validation, exclusive output, server-owned backend configuration, default/mock safety and source+mask handoff. One actual local mock Magic Hour dashboard job produces two saved alternatives with backend/mode recorded and network POST disabled. Initial test assertions incorrectly looked for report.json instead of result.run_dir/manifest.json, then exposed the real generic backend-name gap. Both test correction and backend/mode implementation are explicit; no provider result is inferred.

Visual inspection: actual local homepage screenshot shows the mock model/zero-spend banner and existing upload/template/instruction controls at desktop width. Preview images in that screenshot are deliberately tiny gray test tiles, not generated covers. This verifies the banner/layout only. The native garment review pixels were inspected separately in the preceding checkpoint. No mobile rerun or full live dashboard test is claimed.
