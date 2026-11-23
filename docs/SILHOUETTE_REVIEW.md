# v478: actual silhouette review before browser intersection

Free local CPU only. Existing v477 and all earlier versions are preserved. No default model or automatic inference backend changes. No new model/dependency install.

## Why

The original twenty-source v477 stress test produced severe box-only HQ-SAM failures. A reviewed box does not mean its inferred silhouette is correct. One positive foreground point substantially improved seven of eight selected retry silhouettes, but backpack1 remained badly damaged. This is not a quality-pass rate. See separately delivered original20 evidence with retained failures and source limitations.

## Browser flow

1. Load exact source. Draw object boxes and at least one positive foreground point on every object. Inspect/export the source-bound prompt JSON.
2. Supply the existing source-matched automatic alpha ZIP and click `1. Preview inferred silhouette (CPU)`. This runs pinned HQ-SAM only. No combined result is created.
3. Inspect actual HQ-SAM silhouette on white, black and magenta. Check completeness, holes, background selection, straps and edges. If wrong, edit prompts and retry. No automatic flipping or acceptance.
4. Only after all preview images load can the separate silhouette checkbox be used. Check it only if the silhouette is correct. `2. Accept silhouette and combine alpha` is a second request.
5. Server verifies the frozen source, automatic files, prompts, inferred mask and actual preview image hashes. Changed inputs/previews and replayed confirmation are rejected. It then intersects the frozen reviewed mask with the automatic alpha and returns separate final previews plus a downloadable ZIP.

Changing source, geometry, names, points or automatic ZIP clears pending preview and acceptance. This is source-bound review confirmation, not production quality acceptance. The browser cannot prove a person truly inspected the images. The legacy programmatic run_pipeline helper remains for internal compatibility; the browser endpoint no longer uses it to combine without the separate confirmation.

## Verification

319 tests passed, 1 skipped, 1 intentional duplicate-ZIP fixture warning, 95.28 seconds. Includes twelve new silhouette-stage tests plus confirmation-route same-origin coverage: inference without combination, exact confirmation, repeated confirmation rejected, missing positive point rejected, invalid paths, wrong acceptance/fingerprint, and changed source/mask/preview/automatic/prompts rejected.

Actual Chromium desktop 1280px and mobile layout 390px runs used the real CPU checkpoint and exact numbered-notebook source. Both verified no combination before explicit review, changed ZIP clears acceptance, successful re-inference and confirmation, downloaded actual results, no JS errors, no horizontal overflow and stale prompt edit clears results. The same selected ZIP does not emit a browser change event; the change test uses a differently named ZIP with identical bytes.

Downloaded archives were checked: source RGB exact, min-alpha exact, frozen preview snapshot hashes exact, final output hashes exact and explicit review-state record present. Screenshots of pre-acceptance and final desktop/mobile pages were inspected. This is browser pointer/layout testing, not physical mobile touch verification or owner-PC model execution.

## Still not solved

This does not repair a bad HQ-SAM inference. Positive points do not guarantee quality. Intersection cannot restore missing automatic alpha, uncrop products, recover source-exact fine print independently, create true overhead geometry from a side view, or produce optical glass transmission. The same numbered notebook is an integration example, not broad quality acceptance. Existing twenty-source failures remain. Overall NO-GO remains until the original acceptance requirements are independently met.
