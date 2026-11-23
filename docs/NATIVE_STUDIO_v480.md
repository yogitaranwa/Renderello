# v480: native-size studio candidates, not polished-template acceptance

This additive CLI makes eight designed 2D studio candidates. It does not generate a photographed set, reconstruct lighting, turn the product, complete cropped objects or model transparent glass. Existing default/UI routes are unchanged.

## Run

```sh
python -m renderello.cli native-studio --bundle inspected-opaque-bundle --template all --source-pose front --material opaque --framing complete --regions source-bound-regions.json --out new-output-directory
```

Use `--source-pose elevated` only for an inspected elevated source; no camera change is performed. `--regions` is optional but required for the demonstrated final-scene fine-print checks. A comma-separated subset of template names is accepted. Never call an elevated product physically grounded merely because the CLI runs.

## Fidelity and gates

Input is an RGB-unchanged matting bundle. Source, cutout and alpha output hashes must match its manifest; decoded RGB and alpha must agree. The product is translated once at native scale: no resize, rotation, warp or relight. Every alpha255 product pixel in the final scene is compared exactly against the input. Softer edges mix with the new background and are not claimed exact. An opaque core and background source border are required, but the border check does not prove the photograph is complete.

Transparent, unknown/cropped framing, overhead/unknown pose and products larger than760px are rejected. Narrow plinth and steps reject overly wide product groups. Reusing an existing output directory is rejected to preserve earlier attempts. Source-bound print regions carry original source hash and dimensions, unique names and integer boxes. Their translated final-scene RGB, alpha255 support and output hashes are checked. This is retention of photographed pixels, not independent transcription, legibility certification or regulatory validation.

## Actual verification, November10 2026

- Core suite:378 passed,1 skipped, one expected duplicate-ZIP test warning,100.04 seconds. Thirty new tests. No new install, paid API, GPU session or MagicHour use.
- Actual CLI: eight candidates on a genuine500x667 photographed deodorant back label. All eight have exact opaque product cores and68,036 pixels across three declared print regions unchanged and fully opaque. Output hashes verified independently. Source resolution remains500x667, not the original high-resolution photograph.
- A second product, the inspected elevated backpack/device group, ran through porcelain, amber and noir. Exact opaque cores and output hashes verified. Visually the backpack can appear to float because a fixed elliptical shadow does not recover physical contact geometry.
- All eight deodorant candidates and the three backpack candidates were inspected as actual pixels in contact sheets; full clay and noir images were inspected too. First-attempt plinth/steps contact gap was retained; support was corrected using alpha>127 baseline placement. Scenery remains intentionally illustrative. Bright source illumination looks less natural on dark scenes. This is not eight accepted polished templates.

## Templates

`porcelain_sweep`, `sage_arch`, `clay_plinth`, `noir_halo`, `amber_paper`, `linen_daylight`, `cobalt_aperture`, `blush_steps`.

## Evidence

The separate evidence ZIP contains final candidates/manifests, native product crops, background and shadow layers, independently checked verification JSON, logs, actual source-bound bundles/regions, source/license records, and preserved first-attempt contact-gap images. Contact-sheet thumbnails are reduced for inspection; only full candidate PNGs are used for exact-pixel claims.

Overall acceptance remains NO-GO: the original20-source quality requirement, transparent optics, complete-source needs, independent fine-print validation, eight polished templates, specialist blind comparison and owner-PC execution are not closed by this update.
