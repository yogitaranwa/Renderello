# Explicit IS-Net CPU comparison
10 November2026. Version0.4.71. Overall acceptance remains NO-GO.

## Result
IS-Net general-use removes the large orange slab between headphone earcups/headband that the earlier U2NET mask retained, without manual trimap corrections. The final source-sized magenta/white/black previews were inspected. Warm orange fringes/reflections remain, and the source cable is cropped. This is not a complete-product or clean-edge pass. The yellow notebook still loses its cover and most of its pen, so it fails semantic completeness.

Do not change the default model based on two examples. Both models can omit major product parts. A source-exact RGB cutout can still be visibly wrong because its alpha is wrong.

## Added optional route
```
python -m venv .semantic-env
.semantic-env/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-semantic.txt
.semantic-env/bin/python -m renderello.cli semantic-mask --source original.png --out new-attempt --allow-download
```
Windows interpreter path is `.semantic-env\Scripts\python.exe`; owner-PC operation is NOT tested. Existing UI/default backend remains unchanged. No paid API is used. No automatic fallback to rembg's newer default or BRIA models is allowed: model is explicitly `isnet-general-use`, CPUExecutionProvider only. Package is pinned to rembg2.0.69. Advertised2.0.87 was not available from the install registry, so the publisher-checked2.0.69 release was used instead. Publisher JSON and install log retained.

The optional first download is179MB, from the official rembg GitHub release. Model conversion is maintained by rembg, not a claim that the original IS-Net author provided this ONNX. MD5 checked against installed official rembg session code: fc16ebd8b0c10d971d3513d564d01e29. Observed SHA25660920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a. Checksum disabling is removed from the subprocess environment; mismatched existing models refuse rather than silently overwrite. Model cache stays in Renderello/.semantic-models. Cached runs need no download flag. Weights/environments are excluded from the code ZIP.

Each image runs in its own subprocess with180s wall-time and a Linux /proc2.5GiB RSS guard. Windows has the wall-time guard but not this RSS check. A four-million-source-pixel cap refuses large inputs without resizing. A multi-image exploratory process was killed after saving yellow; its cause was not verified. The two final app CLI cases each completed in separate subprocesses. Failed earlier evidence remains, not silently discarded.

Outputs: original decoded RGB, estimated alpha, RGBA, black/white/magenta previews, output hashes and source/model/package provenance. All original RGB values are retained before later compositing. No pixel-fidelity claim about post-resize output. Quality, semantic completeness and source framing remain NOT_VERIFIED/NOT_EVALUATED. Inspect all parts and holes. This is mask generation only; it does not estimate foreground colors, fix background reflections, relight, restore cropped cables or prove camera/contact correctness.

Use its source-sized `estimated-alpha.png` as input to the optional matting stage only after inspecting semantic completeness. Do not run a complete-product composition on the cropped headphones by declaring it complete. No UI control was added for this CLI stage.

## Verification
- Full current local CPU suite:250passed,1skipped94.63s.
- Four new semantic-mask tests cover explicit download, no-overwrite checksum rejection, exact original RGB/explicit session/provenance, existing-attempt protection, source pixel cap.
- Final app semantic-mask command executed separately for real headphones and yellow notebook; source RGB unchanged and model hash recorded.
- Same source-sized U2NET versus IS-Net review figure inspected. Final headphone white/black previews inspected. Improvement is bounded to removing the inner orange slab, not general quality.
- Kaggle prior version10 remains the only observed committed version. Two import attempts froze the editor, even after reduced203KB input and reload/fresh-tab recovery. No Run/Save&Run clicked, no GPU submitted. Draft import state unverified. Before import,0ActiveEvents and draft session off were verified. Lease released. No active monitor.

## Source basis
- https://github.com/danielgatis/rembg : MIT package, explicit model sessions, IS-Net conversion. Latest docs warn model licenses differ from library license; no BRIA model was used.
- https://pypi.org/project/rembg/ : publisher/project links checked. Exact2.0.69 registry metadata fetched through official PyPI JSON.
- https://github.com/xuebinqin/DIS : original official IS-Net/DIS repository, Apache2.0 code, optimized general-use model announced August2022. DIS2.0 unreleased status in current README must not be confused with this released general-use model.
- https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx : checksum-verified official rembg conversion used here.
- Actual sources: https://www.pexels.com/photo/vibrant-yellow-notebook-and-pen-flat-lay-35826069/ and https://www.pexels.com/photo/black-headphone-on-orange-background-20385205/ .

Next useful test is a source with complete framing and similar thin structures, plus a genuinely different semantic method for same-color product/background failure. More edge erosion cannot restore the missing yellow cover. No claim of superiority across20sources, fine print,8templates or specialists follows from this test.
