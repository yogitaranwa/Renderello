# v479: recover opaque support, check label opacity

## Reason

Min(automatic alpha, reviewed mask) cannot restore any cloth the automatic alpha already erased. An explicit separate opaque route can use a reviewed semantic silhouette as support instead. This is not automatic acceptance or a union of unreviewed guesses.

## Added CLI commands

`opaque-recover --source PHOTO --reviewed HQ_BUNDLE --reviewed-alpha-sha256 HASH --material opaque --framing complete --confirm-reviewed-silhouette --out NEW_ATTEMPT`

Verifies exact source hash, dimensions, source/cutout RGB, binary HQ-SAM mask, pinned model hash, prompt hash/origin/selected object names and a positive point per selected object. The supplied alpha hash binds the exact inspected silhouette. Uses existing guarded closed-form or large-kernel matting with a heuristic narrow trimap and `edge_colors=False`. No RGB estimation, no automatic saliency alpha locked as background. Rejects unknown/transparent material, cropped/unknown framing, reused output and missing confirmation. Material/framing and actual inspection are caller declarations, not automatically proven facts. Existing ninety-second/1500MiB matting worker guard remains.

`label-retention --source PHOTO --cutout SOURCE_SIZED_RGBA --regions SOURCE_BOUND_JSON --out NEW_REPORT`

Regions file contains source_sha256, image_size and regions with unique name, integer box[x1,y1,x2,y2] and optional minimum_alpha(default255). Checks RGB equality AND opacity in the declared source-coordinate rectangles. A cutout can store exact RGB while making all lettering invisible with alpha0. This diagnostic catches that. It does not read text, verify its correctness, fix blurred print or certify later resampled scene outputs. It rejects resampled/different-sized cutouts.

## Actual free CPU tests

348 passed, 1 skipped, 1 intentional duplicate-ZIP warning, 95.13 seconds. Twenty-nine new tests cover provenance, material/framing/confirmation gates, alpha hash, model/prompt/source/cutout mismatch, positive points, binary mask, replay/preservation and independent RGB/opacity retention failures.

Three real-source CLI trials use existing pinned HQ-SAM tiny and PyMatting1.1.16, no new packages or checkpoints:

- `tshirt2`: six product foreground points plus five background negatives recover black fabric and original printed lettering. Earlier positive-only trial had a lower-right background artifact; retained, then removed with explicit negative points. Existing saliency output loses opacity in all47,675pixels of three declared shirt interior regions. Recovered output has all47,675pixels source-RGB exact and alpha255. This is retention, not a specialist quality rating.
- `backpack1`: five foreground points recover main bag plus explicitly selected device group after box-only and single-point failures. Final source-RGB exact, edges still reviewed as research output, not automatic success.
- `deodorant_backlabel`: existing500x667photographed back label. Three ingredient/direction, batch/MRP and contact regions all have source-RGB exact and alpha255. It retains photographed tiny print; no independent transcription or regulatory correctness is claimed. Source isCCBYSA4.0and attribution travels with evidence.

All final cutout RGB pixels and every output hash were checked. Final white/black/magenta product images were visually inspected; comparison boards are evidence. Old failures and intermediate masks remain.

## Research

See separate evidence research/report.md and sources.md: officialSAM2andPyMatting, garment pipeline documentation, TransMatting/paper, TOM-Net/paper and two Reddit discussions. Transparent-460restrictscommercial use/redistribution, so no dataset import. TOM-Netusesattenuation/refractiveflow and an oldTorch/CUDAstack; not installed. Glass remains unsolved, not relabeled as an opaque cutout.

OverallNO-GO: no twenty-source quality acceptance, true glass transmission, uncropping, eight polished template acceptance, independent tiny-print review, two blind specialists or owner-PC verification. This additive version preserves all older versions and makes no default pipeline switch.
