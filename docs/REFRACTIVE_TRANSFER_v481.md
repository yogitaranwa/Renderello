# v481: clean-plate refractive difference experiment

Experimental CLI only. No default/UI change. No learned optics estimator, third-party code, model or dataset is bundled. Actual evidence is analytic known flow, not a real glass photograph. Overall NO-GO remains.

```sh
python -m renderello.cli refractive-transfer --source source.png --plate plate.png --target target.png --flow flow.npy --transmission transmission.png --support support.png --spec optics-spec.json --confirm-assumptions --out new-result
```

Inputs: matching uint8 RGB source/old empty background/new background; float HxWx2 NPY flow in source-pixel-to-background dxdy pixels; RGB0..255 per-channel transmission; L0/255 binary object support. Maximum4MP images and80MB flow file. The JSON spec schema is `renderello-refractive-transfer-v1`, color_space `srgb`, flow_convention `source_pixel_to_background_dxdy_pixels`, image_size `[width,height]`, material `transparent`, framing `complete`, map_origin `synthetic_known` or `external_reviewed`, capture_alignment_reviewed/maps_reviewed true, and input_sha256 with exact source/plate/target/flow/transmission/support hashes. See evidence/known-flow-input/optics-spec.json for a complete example. An existing output directory is rejected.

The equation is evaluated in linearized sRGB: new source = source + transmission*(sampled new plate - sampled old plate). Bilinear correspondence sampling is explicit. Negative residuals beyond the declared tolerance, out-of-plane sampling and target gamut violations reject instead of fabricating borders or clipping. Zero-transmission opaque patches are copied exactly. Transparent pixels necessarily change with a new background. The fixed residual cannot model new reflections, full3D optics, changed camera/illumination, caustics or unseen surfaces. A plate/hash alone does not prove physical alignment; explicit review remains necessary.

Validation:26focused tests;404passed1skipped full suite in101.36s, one intentional duplicate-ZIP warning. Actual CLI analytic fixture output equals the expected equation pixel-for-pixel, protects2700 opaque-patch pixels and all outside-support target pixels;12105pixels differ from zero-flow control. The board was inspected as actual pixels. This is not a photographed glass, not a learned flow result, not performance evidence on the ownerPC, and not transparent-source acceptance. No installs/spend/GPU sessions.

Research:10fetched sources, exact URLs in evidence/research/sources.md. Snellcaster is text-prompt synthesis (GPLv3 and separate component terms), RCTrans requires multiple calibrated captures, BackgroundMattingV2 requires an additional plate, and TOM-Net models mask/attenuation/flow with known limitations. No code/weights/datasets were imported. The experiment's source is original code implementing the documented limited equation.
