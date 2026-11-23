# Lucida alpha trial

Opt-in, isolated alpha research, not default segmentation, optical reconstruction, or a glass quality pass.

Publisher model: https://huggingface.co/egeorcun/lucida
Code: https://github.com/egeorcun/lucida
Model revision: `6cbedc9722652dc9a3df91dd871f0c4f3334e922`.
Weight: published v7 `model.safetensors`, 884,878,856 bytes, SHA256 `2f1aa6913426537d4b93dd5f7138ae5c6664e99abda98c95f3d5b9101283e7d5`.

The publisher declares MIT for model/code. Its model card acknowledges research-only training datasets. This is not independent training-data rights clearance. No training dataset is included or downloaded.

`renderello.lucida_mask` verifies exact pinned local publisher code/config/weights and uses safetensors. It never downloads or installs anything. It requires CUDA, avoids implicit CPU fallback, and provides a separate 180-second subprocess guard. It saves source-size alpha, source-exact RGB in a straight-alpha RGBA PNG, and white, black and magenta review composites. All outputs are marked NOT_EVALUATED and NOT_VERIFIED. Default UI and pipeline are unchanged.

Prepare the separately reviewed publisher files inside an isolated Renderello model folder, including an empty `__init__.py`. Keep downloaded weights outside source-code archives. Call `guarded_file` from Python for bounded execution. The direct module entry point is the worker, not the watchdog.

Lucida estimates an alpha mask. This does not infer refraction, transmission, the hidden clean background, a complete product outside the source crop, or remove the old background already visible through glass. Alpha improvement cannot be reported as realistic glass replacement without inspecting a new-background result.

Sixteen deterministic adapter tests cover source-size validation, invalid/empty alpha, exact source RGB, opaque compositing, pinned-file checks, symbolic file rejection, output preservation, and worker failure/timeout. They do not stand in for a real model run.

## Actual execution

The four-source pinned publisher trial completed on Tesla T4 in 55.79 seconds including download/setup. The exact adapter was then exercised through `guarded_file` on the same shirt in 10.17 seconds after correcting the test harness's PYTHONPATH. All six adapter PNGs matched the independent publisher-trial PNGs byte-for-byte. Initial dependency preflight found all modules already present; no installation was needed. One initial CPU-only session was stopped and verified off before GPU retry. The final GPU session was stopped and verified off after output recovery.

Full core suite: 420 passed, 1 skipped, one intentional duplicate-ZIP warning, 101.10 seconds.

Visual result: soft alpha retains glass rims/engraving and shirt cloth/print. Engraved glass retains old room colors and table reflection; faint exterior remnant remains. Stemmed glass and bottle remain source-cropped. No realistic-optics quality pass or overall acceptance.
