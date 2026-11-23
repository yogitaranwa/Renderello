# Explicit seeded mask correction

Use only when a person has reviewed the source and drawn correction regions.
This is manual assistance, not a better automatic segmentation claim.

Source-sized 8-bit grayscale seed PNG values:
- 0: definitely background
- 64: probably background
- 128: leave initial classification unchanged
- 192: probably foreground region
- 255: definitely foreground stroke/region

All other values, wrong dimensions, RGB maps, absent definite foreground/background
seeds, existing output directory, more than4millionpixels or iters outside1-8
are rejected. Never resize a misaligned correction map silently. Thick foreground
strokes alone can leave stroke-shaped fragments when the initial model omitted
a whole strap: draw a probable-foreground region covering the missing object too.

```sh
python -m renderello.refine_mask --source path/to/source.jpg \
  --alpha path/to/initial-alpha.png --seeds path/to/seed-map.png \
  --out path/to/fresh-correction
python -m renderello.cli run --source path/to/source.jpg \
  --templates clean_studio --routes cutout_compose --variants 1 \
  --mask path/to/fresh-correction/refined-alpha.png
```

No model call/download/install. OpenCV GrabCut uses source colors plus explicit
seed regions. Keeps initial alpha and seed PNG, source-RGB cutout, refined alpha,
green review preview and a source/alpha/seed hash manifest. A3x3blur softens the
binary boundary; definite seed pixels are restored exactly. This is not learned
hair matting. Inspect fur, glass, light borders, removed product and frame cuts.

Actual watch1 test: automatic U2NET and U2NETP lost both straps. Sparse strokes
alone failed and were rejected. Explicit engineer-drawn probable-foreground strap
regions plus definite foreground/background strokes recovered both source straps.
Edge jaggies/light metallic borders remain, and the source itself cuts straps at
its frame. No universal mask improvement or human quality acceptance established.
No raw original GPU alpha was recovered. Keep correction sidecar next to alpha
for source/hash-bound ASSISTED provenance during composition/gallery display.

## Source-frame cuts and source records (v0.4.43)
A corrected mask cannot invent missing source pixels. On the watch1 example,
cutout_compose floats the source's straight strap ends inside the new canvas.
This is not publish-ready. The existing scene_compose frame-preserving route
keeps the ends against the canvas edge, but may crop or enlarge the product and
still needs eye review. Mock scenes are not GPU scene-quality evidence.
Both routes now flag source-frame contact and cap the structural score at 2.
The warning is a tripwire, not a claim that all flagged images are bad.

Run manifests include content-bound source provenance. The 24-image registry
uses exact file SHA256, not a filename or path. Renamed identical uploads match;
resized or changed copies are UNKNOWN. The gallery displays the classification,
attribution, owning URLs, hash and rights caveat on the source card itself.
The tshirt1 source is an illustration, not a product photograph. Other pack
images have photographic appearance; their capture origin is not independently
verified. Recorded rights metadata and historical page checks are not current
legal clearance or permission to republish. Legacy runs without a source record
are displayed as UNKNOWN, not silently reclassified.
