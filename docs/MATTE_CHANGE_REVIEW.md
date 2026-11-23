# Native matte-change audit checkpoint

13 November 2026, 00:57 IST. Addendum to the candidate-studio checkpoint. No new commercial acceptance, physical relighting, glass recovery or measured accuracy claim.

## Added implementation

`matte_change_review.py`: 165 physical /151 nonblank lines. OpenCV native connected-component overlap checks expose dropped small parts, newly connected bridges, new isolated foreground, filled original holes and excessive added/removed support. Per-component counts are exact even when detailed JSON is bounded. Bincount and compact label pairs avoid a full-image scan for every component. Strict typed numerical budgets reject bool/nonfinite/out-of-range values. Red/blue maps show byte-alpha increases/decreases, not segmentation error.

Trimap preflight checks missing classes, seed/initial-mask contradictions, unknown allocation budget and components with no definite foreground seed. Initial masks and heuristic trimaps can both be wrong. The audit does not establish which is correct. Uncertainty is preserved, not replaced by a model-agreement accuracy score.

The end-to-end candidate studio is now 293 physical /274 nonblank lines. It saves trimap audits, seed-conflict maps, native matte-change reviews and delta maps. A refined candidate with audit blockers cannot silently become a render-ready mask, even if someone declares its mask reviewed. The initial candidate remains separate. This does not automatically select the initial candidate.

A real execution defect was found and fixed: the guarded matting subprocess could not import its package when called outside the repository. It now explicitly passes its installed/source package root through the child's environment without changing the caller's directory. A real worker test runs from an unrelated directory. The first failed 20-case batch is retained.

New test module:197 physical /153 nonblank lines;42 collected cases. Includes actual guarded execution plus tests for lost accessories, holes, bridges, islands, soft-alpha changes, exact bounds, diagonal connectivity, dense fragmented masks, invalid arrays, missing seeds, heuristic erosion and pipeline blocker propagation. Mocked worker use in one propagation test is explicitly labeled; it is not inference evidence.

Final whole regression after the import fix:1,610 passed,2 skipped,390 warnings,101 test modules in six non-overlapping batches. Focused new audit + candidate studio + matting:153 passed. The test count is not an image-quality score.

## Same20 CPU refinement experiment

Each input is the previously preserved native RGB derivative and BiRefNet alpha. A radius-2 heuristic trimap was created, saved and labeled as not ground truth. The actual guided worker ran at native size with source RGB retained.17/20 executed with exact source foreground RGB;3 failures were preserved. Tiramisu and multiproduct barber were refused by the1MP matting cap. Sneaker's failure was unavailable memory monitoring; do not relabel it as a model failure or successful inference. No automatic fallback, resize, retry or acceptance occurred.

Across the17 executed revisions, the audit flagged5 filled-hole cases,3 damaged-part cases,3 new-disconnected-component cases and1 bridge case. Cases overlap. These are change warnings against an unverified baseline, not proven semantic errors. Three of the20 heuristic trimaps had initial parts with no definite foreground seed. Initial and refined masks remain separate.

Visual spot checks only: native can2 white-background review, teddy magenta-background review, glass1 alpha-change map and watch2 alpha-change map were inspected. The can still has a white rim; teddy still has a hard base and edge defects. No image was promoted. The other13 executed outputs have not received native visual review in this checkpoint. This is CPU guided refinement, not GPU inference or the requested post-code free-GPU comparison.

## Delivery contents and remaining work

Full native outputs are retained locally. This addendum archive includes all per-case JSON reviews, worker manifests, logs, heuristic trimaps and change maps, plus selected native review images and the current complete source/tests snapshot. It does not duplicate every native review image. The earlier archive remains unchanged.

Remaining: inspect every executed native revision independently, investigate the sneaker memory-monitoring failure without weakening guards, then choose a justified free-GPU same20 experiment for an actual model stage. Do not repeat deterministic diagnostics merely to say a GPU was used. Transparent optics, missing semantic objects, true reflection matching and physical relighting remain unsolved. Overall completion remains45% assistant judgment, not a measured percentage. Canonical PDF unchanged; no paid spend.

Open-source sources used here are the already-vetted OpenCV package and existing guided worker. Publisher/license notes and exact tested dependency versions are in docs/CANDIDATE_STUDIO.md and requirements-candidate-studio.txt in the code archive. Primary OpenCV source: https://opencv.org/license/ and https://pypi.org/project/opencv-python-headless/4.11.0.86/
