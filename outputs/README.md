# Existing results, checked against the original brief

Updated 21 November 2026. This is an existing-results handover, not a new generation run or acceptance certificate.

## Start here

- **00-best-existing-review-candidates/**: a small first-look selection. These are the stronger saved visual candidates and the latest saved Magic Hour display, not approved listing photographs. Source images are beside the native 13 November examples. Read `SELECTION_NOTES.md` before using them.
- **01-procedural-benchmark-480/**: the full 20 November diagnostic gallery, 24 source cases x 10 template names x 2 seeds. Each source folder contains its original image, 20 PNG outputs, a ten-template contact sheet and run details. The outputs are deterministic cutout/composites, not Magic Hour generations. All 480 encoded output hashes differ.
- **02-magichour-model-comparison/**: nine actual native Magic Hour outputs, three matched inputs x three models, from 11 November, plus exact API inputs and a contact sheet. Exact prompts, historical costs and settings are included separately. This is a historical comparison, not proof that the current models/prices are unchanged.
- **03-specialist-comparisons/**: three paired comparison sheets and six actual specialist results. Pixelcut has two native exports with different aspect ratios and one partly obscured screenshot. Z Product AI has three screenshots, not native exports. These are not independent human ratings or proof of superiority to two leading specialists.
- **04-nov21-notebook-displays/**: exact embedded PNG displays extracted from the saved notebooks. The CPU display and latest Magic Hour source/result display are labelled separately. They are display evidence, not recovered full-resolution output files. The full 5,087-byte live-run manifest was lost and has not been recreated.
- **05-historical-baseline20-scoring/**: all 20 original baseline PNGs and the existing seven-dimension visual scoring sheet. Ratings are development-side judgments awaiting human review, not independent human ratings. Two were tentative 4/5 usefulness candidates; that did not establish acceptance.
- **06-magichour-scenes-and-composites/**: actual generated empty scenes and four saved original-product composites, 11 November. Product RGB was composited, not generated end to end.
- **07-nov13-native-comparisons-and-failures/**: the 13 November source/product comparisons, including whole-image Magic Hour edits and source-composite failures. None was accepted as a polished exact-product result.

## What the filenames mean

`SOURCE` or `source-original` is an input, not a generated result. `PROCEDURAL` is a saved diagnostic composite. `MAGICHOUR` identifies a real model output or a clearly labelled generated-scene composite. `EMBEDDED-DISPLAY` is an image preserved inside a notebook. `ALTERED-DETAILS` and `SCREENSHOT-LIMITED` mark material limits, and `DEVELOPMENT-SCORING-PENDING-HUMAN-REVIEW` marks development-side scores that await human review. These markers are material limits, not optional footnotes.

`BENCHMARK_OUTPUT_INDEX.csv` maps every one of the 480 diagnostic outputs to source, template, seed, method and SHA-256. `BRIEF_COVERAGE.csv` distinguishes saved artifacts from unmet quality/evaluation requirements. The procedural diagnostics CSV is an automated proxy, not the requested complete seven-dimension photographic scoring of all 480 outputs.

## Notebooks

At the handover root, `Renderello_Studio.ipynb` ships clean: all 185 code cells have empty outputs and null execution counts, with cell source unchanged. In `../docs/`, the saved Magic Hour notebook retains outputs but has null execution counts. It is evidence of the verified run, not sequential full-Run-all proof.

## The brief is not fully passed

The gallery retains weaker results and failures rather than discarding them. Ten configured template names and ten diagnostic backdrops do not establish eight polished, genuinely distinct production-candidate photoshoots. The 480-case set is not a Magic Hour benchmark. The source set includes CGI/mockup cases; capture-origin notes are historical and are not fresh rights clearance. The item named `skincare_tinytext1` is an amber-dropper source, not a verified hard tiny-text skincare test. A separate low-resolution deodorant back-label stress input exists in the Magic Hour comparison, but its generated print is not verified faithful.

The 80% no-critical-alteration and 70% publication-usefulness gates remain unestablished. Practical portrait/landscape generative quality, an accepted best-quality/value pair, complete retry-inclusive economics and an independent human comparison against two leading specialists are also not established. No fresh paid generation or quality acceptance was performed to make this folder.

