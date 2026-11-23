# Source-bound prompt editor, version 0.4.73
10 November 2026. Additive local UI work. Overall Renderello acceptance remains NO-GO.

## What changed
The local home page links to `/mask-prompts`. Upload a source, draw a box for each object, name objects, add positive/negative points, undo edits, remove objects, and explicitly review before exporting JSON. The editor offers fit width, 2x, 4x, and drag-to-pan. Prompt JSON is tied to original file SHA256 and decoded source dimensions. A changed source clears the previous preview, review checkbox and download link. Edits invalidate the previous export. Duplicate names or missing boxes are rejected at export.

Run the existing local server from the project directory:
```
python -m renderello.webui --port 8765 --out runs
```
Open the printed local address and choose "Prepare reviewed mask boxes / points". This page needs only core dependencies. It does not download or run a segmentation model. Original upload bytes and RGB preview are saved in a new `prompt-*` folder under the chosen output directory. The 25 MB request and 4 million pixel caps remain explicit; the editor does not silently resize the source. Keep the original source file unchanged.

Choose Draw box and drag around a whole object. Choose Positive point to mark a product part, or Negative point to exclude an empty hole or background. For small parts, zoom and use Pan, then return to an annotation mode. Add another object for an accessory such as a pen; combined masks are an explicit union, not automatic accessory discovery. Check the review checkbox only after inspecting the geometry. Download reviewed-prompts.json.

Run the exported JSON through the existing optional CPU command after following [its setup](PROMPTED_MASK_v472.md):
```
python -m renderello.cli prompted-mask --source ORIGINAL_PHOTO --prompts reviewed-prompts.json --checkpoint sam_hq_vit_tiny.pth --out NEW_ATTEMPT
```
Inspect all generated masks and black/white/magenta reviews. This editor prepares geometry only; inference is still a separate CLI step. Default automatic generation remains unchanged. No paid service or GPU is used.

## Verification
- Final clean core suite: **265 passed, 1 skipped in 94.28 seconds**. Six new tests cover exact preview/source binding, unique output folders, bad inputs, pixel cap and actual GET/POST routes. Final test log is included.
- Real Chromium browser test at widths 1280, 390 and 320: source upload, two boxes, one negative point, review/export and actual file download. Downloaded coordinates matched exactly at all three widths. Zoom/pan, undo, object removal and source-change invalidation also executed. No JavaScript page errors or horizontal document overflow were observed. These are automated pointer tests, not human usability ratings or physical-phone touch tests.
- Downloaded 320-width JSON was passed to the real optional Light HQ-SAM CPU CLI using the exact yellow source and pinned checkpoint. Two object masks and combined review images were produced. All saved RGBA RGB pixels exactly match decoded source RGB. This verifies editor-to-CLI format compatibility, not mask correctness.
- Actual desktop/mobile screenshots, mobile detail crops, zoomed preview and real mask review were inspected. Short mode labels fit the smallest tested width. Layout remains a plain vertical preparation form, not a finished production UI.

## Remaining limits
Desktop review is recommended for precise annotation. Physical touch input, owner Windows ARM PC, and model inference from the UI are unverified. Browser preview and CLI deliberately use the same PIL RGB decode without EXIF transpose.

The tested yellow mask recovers the cover and separate pen. Ring boundaries and pen-tip/cap edges still have visible defects. Source color at edges and incorrect topology remain possible. Model scores do not prove completeness. Reviewed geometry was selected by the assistant; it is not a user or independent specialist rating. Optional-environment full-suite completion remains unverified from version 0.4.72; only the clean core suite is reported passed here. The original 20-source acceptance, genuine fine print, eight polished templates, two-specialist blind comparison and owner-PC acceptance remain open.
