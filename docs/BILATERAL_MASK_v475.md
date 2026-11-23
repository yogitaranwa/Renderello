# BiRefNet-lite automatic mask experiment, version 0.4.75
10 November 2026. Overall Renderello acceptance remains NO-GO.

## What improved, and what did not
BiRefNet general-lite automatically recovers the yellow notebook cover and separate pen that U2Net, IS-Net and InSPyReNet fast omitted. No box or point prompts were used. It still keeps a large yellow background bridge between the objects. That error is conspicuous on the magenta review. This is an automatic same-color improvement, not a correct final mask.

On the complete silver scissors source, both handle holes open automatically and the edges look smoother than the earlier reviewed HQ-SAM attempt. Fine neck detail and gray/white edge fringes still need review. There are no independent human ratings, no complete 20-source evaluation for this model, no tiny-print acceptance and no general quality claim.

## Added
The optional explicit `bilateral-mask` CLI uses the rembg-maintained ONNX conversion of BiRefNet general-lite. It does not claim those bytes are the original author's PyTorch checkpoint. Only the tested SHA256 is accepted. CPUExecutionProvider only, no prompts, no downloads, no default model/UI changes. Original decoded RGB is used byte-for-byte for saved cutouts. Internal model input is 1024x1024; the saved source dimensions are unchanged and alpha is resized to match them. Dichotomous segmentation is not optical glass matting.

The wrapper rejects missing/wrong checkpoints, existing output folders, source images over 4 million pixels, invalid/empty alpha, constant or nonfinite predictions. Worker wall limit is 180s per source, Linux-only sampled RSS guard is 8 GiB. Windows does not have the /proc RSS guard. A system can kill the worker before the guard: this happened in the smaller research environment. `infer_file` called directly bypasses worker limits. Do not treat a killed attempt as a pending/successful result or retry it unchanged.

Graph optimizations, CPU memory arena and memory pattern are disabled; two intra-op threads and one inter-op thread. This configuration was tested on free Kaggle CPU, not GPU. No paid service/credits were used. Owner Windows ARM PC execution and memory sufficiency are unverified. Keep this as an explicit alternative rather than switching an unattended default.

## Setup
Use a separate project environment, core requirements plus:
```
python -m venv .bilateral-env
.bilateral-env/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-bilateral.txt
```
Optional runtime pinned to onnxruntime 1.31.1. Official registry metadata and publisher links were checked. The Kaggle CPU environment was Python3.13.18, NumPy2.1.3. Owner PC uses `.bilateral-env\Scripts\python.exe`; it has not been tested there.

Download the exact conversion as a file, not a script:
https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx
224,005,088 bytes. Publisher MD5 `4fab47adc4ff364be1713e97b7e66334` verified. Wrapper SHA256 `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`.
```
.bilateral-env/bin/python -m renderello.cli bilateral-mask --source original.jpg --checkpoint birefnet-general-lite.onnx --out new-attempt
```
Inspect source and all black/white/magenta reviews; a mask with an object present can still have wrong empty spaces and wrong accessories. Background reflection/color remains inside unchanged RGB. Source-cropped parts cannot be recovered.

## Verification
Clean core suite: **283 passed, 1 skipped in 97.01 seconds**. Eleven new tests cover checkpoint/source validation, alpha/logit failures, output preservation and RGB/provenance using mocked inference. Core tests do not count as model-quality evidence.

Initial local CPU probes on yellow/scissors were killed with signal9 around1.84GB sampled RSS, without masks. A materially changed direct-ONNX probe also died. Cause unverified. The local runtime then was1.23.2. A fresh private Kaggle CPU notebook with direct cell entry worked around an older notebook's frozen import dialog. The first research probes completed in39.95s/39.47s. ZIP1,663,526bytes and SHA256 `c4a23c3eac464c8e03d19560948f1b7b9f9e59c4ccd84bce9feb64418baffe73` were checked after download. Saved RGBA RGB arrays were checked exactly equal to decoded original bytes, and source hashes matched. Actual magenta reviews and matched source-coordinate comparison were inspected. Its draft session stopped,0activeevents confirmed, and browser released.

The exact application module verification is recorded separately in the accompanying evidence. Do not infer owner-PC or full UI inference from a research notebook.

## Sources
- Official author code, MIT: https://github.com/ZhengPeng7/BiRefNet and https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE . Localization/reconstruction and bilateral-reference approach, not proof of our results.
- Primary paper: https://arxiv.org/abs/2401.03407v6 . Bilateral references use image patches and gradient guidance. Published benchmark claims are not a Renderello acceptance pass.
- Official author's lite model: https://huggingface.co/ZhengPeng7/BiRefNet_lite . Model metadata declares MIT and links to the author repository. Metadata revision aa62cd87eafb9cc43056d08ef3615a14628b831d retained. No remote-code trust loader was used.
- Conversion/distribution: https://github.com/danielgatis/rembg . Installed rembg2.0.69 session file supplies exact conversion URL, MD5 and1024 preprocessing. Our wrapper reproduces those input normalization/resize choices, with finite/constant prediction guards added. Latest rembg documentation is not assumed identical to installed2.0.69.
- Official runtime metadata: https://pypi.org/pypi/onnxruntime/json . Microsoft project link verified.
- Exact private test notebook: retained in the private evidence bundle (Kaggle CPU-only run, Nov 10). Code/source bytes and logs retained in the private evidence bundle.
- Yellow source: https://www.pexels.com/photo/vibrant-yellow-notebook-and-pen-flat-lay-35826069/ . Scissors: https://www.pexels.com/photo/silver-scissors-on-white-surface-3990146/ . Source hashes bind comparisons to unchanged bytes.

Keep explicit model alternatives and the reviewed prompt fallback. Automatic same-color cover retention has improved here, but the retained bridge is still a real failure, not a detail to hide.

## Exact application-module run, 18:42-18:46 IST
The exact shipped `bilateral_mask.py` SHA256 `ed82f4d4149ef179402d025fb3fc4574b12087f4921606ed76195ba79c44f57b` was embedded unmodified in a new private Kaggle CPU cell and called through `guarded_file`, not the earlier research probe. Both real images completed:41.85s yellow and41.33s scissors,95.43s whole cell including setup. Both APP_CASE_END markers and module hash were recovered. Downloaded ZIP1,610,239bytes SHA256 `f380b4dc1ac6bc35eb4612afa4fe7d4421f5435cc2a8c7eeb94161d56e0d569a` matched the execution output. Rechecked source hashes, every saved output hash, and exact decoded source/cutout RGB arrays for both cases. Actual white/black/magenta pixels inspected for both images: same yellow bridge failure; scissors holes open with fringe/neck-detail caveats. This verifies the explicit worker path on Kaggle CPU only, not end-to-end inference through the web UI or owner PC.

Second draft session stopped at18:46IST. Session off and0activeevents were verified before releasing it. No background session remains from this test.
