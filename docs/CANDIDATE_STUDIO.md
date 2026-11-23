# Candidate studio

Checkpoint: 13 November 2026, 00:48 IST. Research software, not production photo acceptance.

## Run

Use a project virtual environment and install `requirements-candidate-studio.txt` from the official PyPI index. This does not include a segmentation checkpoint. BEN2/BiRefNet masks can be supplied as existing separate candidates; an optional local BiRefNet checkpoint uses the existing guarded worker. Do not download weights merely because a name looks familiar. Consult `MODEL_LICENSE_REVIEW.md` and the checkpoint's own license.

```
python -m renderello.cli candidate-studio \
  --source source.png --alpha BEN2=ben2-alpha.png \
  --alpha BiRefNet=birefnet-alpha.png --out fresh-diagnostics
```

The default makes native cutouts, dark/white/magenta checks, bounded component/hole diagnostics and disagreement images. It never picks the model that agrees most, deletes small objects or fills holes. A disagreement color map is not an accuracy map. A complete diagnostic directory is not a usable photograph.

For contact rendering, add `--spec reviewed-spec.json --background native-background.png`. A spec must bind the exact normalized source and every alpha using the shape/dtype-aware `candidate_diagnostics.digest` function. `selection` is explicit. It must name a reviewed mask, material, framing, pose, camera, contact spans and review declarations. `actual-probes/*/spec.json` in the evidence archive gives real worked inputs. These declarations record local engineering review, not authenticated user approval.

The optional `background_match` object needs a genuinely reviewed backdrop sample mask, an amount no greater than 0.5, and a maximum change budget. Zero alpha alone is not evidence that a sample is true background. The optional `poisson` object requires `color_change_permitted: true`; its protected mask must be binary and within alpha=255. Poisson changes colors and may introduce seams even when protected interiors are exact. Neither method is physical relighting. The contact-only baseline and changed-pixel map remain separate.

Optional matting needs an explicitly supplied trimap and named candidate:

```
python -m renderello.cli candidate-studio --source source.png \
  --alpha BEN2=alpha.png --matting-candidate BEN2 --trimap trimap.png \
  --matting-method cf --out fresh-matting
```

The older `matting.make_trimap` helper makes a heuristic trimap, not ground truth. Preserve that distinction. Closed-form/large-kernel workers use PyMatting; guided mode uses the existing OpenCV path. Refined masks remain separately named. A 48x48 real closed-form smoke and a real guided full-pipeline test ran; this is not validation of native closed-form matting on all 20 photographs.

## Enforced limits and honest failures

- Native uint8 RGB, minimum 3x3 and maximum 4 million pixels. No silent downscale. Extremely small or odd-aspect inputs get review flags.
- EXIF orientation is normalized before mask pairing. All eight EXIF orientations are covered by intake tests. Explicit color conversion is required for ICC profiles. Embedded alpha is retained and a candidate cannot reveal pixels beyond it.
- Single-frame L alpha at exact normalized size. Refuse RGB/16-bit/palette/animated guesses, transformed masks and invalid paths. Preserve original inputs.
- Components, holes, dark/white foregrounds, large soft bands and frame touches are descriptive flags. Multi-object parts remain intact. Explicit include/exclude regions can reject a missing desired object or unwanted prop; the code does not know semantic intent on its own.
- Upright contact requires reviewed lower-contour spans. Multiple spans must not overlap or bridge gaps. Flatlay uses finite outside-distance support. Cropped, suspended, incompatible-camera, unknown-material and frame-touching cases are not repaired by inventing missing parts.
- Transparent materials are blocked: alpha blending is not refraction/transmission. Reflective materials are blocked: source lock is not environment reflection matching. Real-glass acceptance requires evidence absent from these same20 inputs.
- Unknown specification fields, changed pixel hashes and invalid numerical settings fail. Existing output directories are never overwritten. Failures after output creation retain INCOMPLETE and failure.json. Intake rejection can happen before any output directory exists; its exception is preserved by the calling batch log.
- Worker failure propagates. The timeout propagation unit test uses a mock; it is not evidence of actual timeout performance. Missing optional dependencies do not silently select another model.
- Markers, per-file hashes and source-bound specs aid replay. They do not certify an image or establish user consent.

## Measured checkpoint

The four new modules contain 823 physical lines, 757 nonblank lines. Tests contain 510 physical lines, 410 nonblank lines and 97 collected cases. These counts include comments/docstrings and exclude installed libraries and downloaded models. Full regression: 1,568 passed, 2 skipped, 390 warnings, across 100 test modules in five non-overlapping batches. See code-census and regression logs in the evidence archive.

Same20 diagnostic run: 20 executions, two candidates each, 40 exact native cutout-RGB and artifact-hash checks. The first original-PNG attempt stopped at item 13 because ICC conversion had not been authorized. That log and its 12 completed directories were preserved. The complete run uses previously recorded, profile-free inference derivatives whose decoded RGB matches the originals. It is not a fresh GPU model run.

Three native opaque contact probes and three opt-in background/Poisson probes were rendered and visually inspected. Shadows remain subtle; source halos and scene realism are unresolved. No new commercial acceptance. The preceding independently reviewed model comparison remained 0/20 commercial acceptance. There is no measured accuracy percentage or ground-truth segmentation score. Overall 45% completion is an assistant judgment, not a measured KPI.

## Open-source sources and licenses

- OpenCV Python package 4.11.0.86: publisher shown as OpenCV Team on official PyPI, Apache 2.0 for OpenCV >=4.5, wrapper/license dependencies retain their own notices. Used for components, distance geometry, guided refinement and optional Poisson cloning. https://pypi.org/project/opencv-python-headless/4.11.0.86/ and https://opencv.org/license/ and https://docs.opencv.org/4.10.0/df/da0/group__photo__clone.html
- scikit-image 0.25.2: official PyPI project links to scikit-image documentation; BSD-3-Clause with per-file BSD-2/MIT notices. Used for background-only histogram matching, not learned relighting. https://pypi.org/project/scikit-image/0.25.2/ and https://scikit-image.org/docs/stable/license.html and https://scikit-image.org/docs/stable/api/skimage.exposure.html
- PyMatting 1.1.16: official PyPI author is The PyMatting Developers; homepage/source match the project repository. MIT. Used by the existing optional closed-form/large-kernel worker. https://pypi.org/project/pymatting/1.1.16/ and https://github.com/pymatting/pymatting/ and https://pymatting.github.io/alpha.html

These dependencies are not counted as authored code. Preserve upstream notices when redistributing them. Checkpoint/model licenses are separate from Python-package licenses. The environment freeze records the tested optional state, not a promise that arbitrary newer releases behave the same.

## Remaining work

A post-code free-GPU same20 model/refinement experiment has not run. Select a concrete matting hypothesis, preserve the same source/native coordinates and both old masks, record heuristic trimap limitations, compare native outputs independently and reject regressions. Do not run a GPU merely to repeat deterministic diagnostics. Real transparent optics, physical relighting, reflection matching and automatic semantic intent remain unsolved. This checkpoint does not change the canonical PDF.
