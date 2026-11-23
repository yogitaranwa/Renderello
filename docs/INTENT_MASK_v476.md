# Reviewed object intent with automatic alpha, version0.4.76
10 November 2026. Still NO-GO overall.

## Result
The new explicit `intent-mask` CLI combines a source-bound reviewed HQ-SAM silhouette with an existing automatic alpha. On two real sources it removes the unwanted numbered props and yellow background bridge without changing decoded product RGB. The yellow notebook and separate pen stay present, and the numbered notebook page/rings stay present. This is manually reviewed object intent, not an automatic segmentation success.

The rule is simply `minimum(automatic alpha, reviewed binary HQ-SAM mask)`. Inside the reviewed foreground, every automatic alpha value is kept exactly. Outside it, alpha is0. No model choice, new generative pixels, RGB correction, silent default or new download. The gate cannot recover pixels missing from the automatic mask. HQ-SAM's hard jagged boundaries can transfer to the combined output; cyan ring fringe and source-color edges remain. This is an intent correction, not a polished edge or glass-matting claim.

## Usage
1. Use `/mask-prompts` to load the exact original. Select every wanted object by reviewed boxes/points and export JSON. It prepares prompts, not inference.
2. Use `prompted-mask` with the pinned free Light HQ-SAM checkpoint and that JSON to produce a reviewed mask folder. Inspect its original, object masks and three background reviews. Box/point intent alone does not prove the inferred silhouette is right.
3. Use an explicit automatic mask command, e.g. `bilateral-mask`, and inspect its reviews. Never substitute a different source crop or resized original.
4. Combine the two folders:
```
python -m renderello.cli intent-mask --source original.jpg --automatic auto-result --reviewed reviewed-result --out new-intent-result
```
No extra dependencies for the intersection beyond core. Model setup/resource caveats remain in `PROMPTED_MASK_v472.md` and `BILATERAL_MASK_v475.md`.

The command verifies exact original SHA256, source dimensions, original decoded RGB and RGBA RGB, stored alpha/cutout consistency and recorded output hashes. Reviewed prompts must bind to the source hash/dimensions and declare reviewed_source_geometry; model hash, prompt hash and selected object names are checked. Reviewed alpha must be binary. Empty intersections, ambiguous automatic manifests, source mismatch, altered output bytes, changed prompts and existing output folders are rejected. It cannot authenticate who actually reviewed the prompts: the origin marker is a provenance declaration, not human approval. All saved results remain NOT_EVALUATED/NOT_VERIFIED.

## Actual execution
- Numbered notebook original1125x750: engineer-reviewed source box[375,95,835,630], no points. Newly executed actual CPU HQ-SAM then intersected with previously verified exact v475 BiRefNet alpha. Removes26,411previously nonzero pixels, including visible unwanted number props. Full original unchanged, no crop. Source is complete, but no overall product acceptance or fine-print pass.
- Yellow notebook original1125x750: exact v473 browser-editor JSON and its actual CLI HQ-SAM result reused, two reviewed objects. Combined with verified BiRefNet alpha. Removes17,237nonzero pixels, including the visible yellow bridge. This is the previously browser-tested editor export, not a hand-recreated prompt. Retained cover/pen and remaining jagged edges verified on actual pixels.
- Exact final CLI ran in the clean core environment without torch or ONNX inference. Verified source RGB byte equality, exact min-rule and unchanged surviving automatic alpha values for both. All actual white/black/magenta reviews and matched source-coordinate comparisons inspected. No paid/GPU action or new installation.
- Twelve new core tests cover exact min/RGB/provenance, output preservation, mismatches/altered prompts/wrong model/nonbinary/empty/ambiguous inputs and inability to recover missing automatic alpha. Full clean core suite **295passed,1skipped in96.92s**. Earlier invocation with the system Python lacked pytest and did not run tests; the isolated test environment produced the result above.

Owner-PC execution, full web-UI inference/combination,20-source acceptance,8polished templates,tiny print and independent specialist blind ratings remain open. The new route is CLI only; the existing prompt editor is its input step.

## Sources and evidence
- Reviewed method: https://github.com/SysCV/sam-hq and https://arxiv.org/abs/2306.01567 . Exact pinned checkpoint in prior v472docs, no change/download in this run.
- Automatic method: https://github.com/ZhengPeng7/BiRefNet and https://arxiv.org/abs/2401.03407v6 . Exact rembg-maintained ONNX conversion in v475docs, not original author checkpoint bytes.
- Original numbered notebook: https://www.pexels.com/photo/notebook-1314540/ . Original yellow notebook: https://www.pexels.com/photo/vibrant-yellow-notebook-and-pen-flat-lay-35826069/ . Source hashes, sourcebytes, prompt JSON, reviewed/automatic input folders, output manifests and comparisons retained in private evidence.
- The intersection rule is this implementation's deterministic design, not a claim from either paper. Published benchmark results are not Renderello's results.
