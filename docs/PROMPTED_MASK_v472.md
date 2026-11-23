# Reviewed prompts for difficult masks
10 November2026. Research version0.4.72. Overall acceptance remains NO-GO.

## Tested result
Free CPU Light HQ-SAM (TinyViT) with an engineer-reviewed box recovers the yellow notebook cover that automatic U2NET and IS-Net omitted. A second box selects the pen, explicitly as another object. Additional negative points remove a background island but introduce a small pen-cap hole. Notebook ring details remain imperfect. This is a manually prompted improvement, not an automatic pipeline success.

A new real, complete-framed overhead silver scissors source was inspected before testing. IS-Net leaves the white background inside both handles. HQ-SAM box-only also fills those holes, despite a high predicted score. Negative points open them. The final inner edge remains jagged; low-contrast blade/neck details are not fully verified. Do not claim a quality pass from either the figure or predicted score. A second gray/green scissors candidate looked like a paper/craft depiction and was excluded from real-product testing.

## Added
`prompted-mask` is an explicit optional CPU CLI, not a change to the default model or UI. It requires a reviewed JSON prompt file bound to the exact original source SHA256 and dimensions. Boxes/points are source-pixel coordinates. Positive labels are1, negative0. Each selected object gets a separate mask; the saved combined alpha is their explicit union. This is not automatic accessory discovery. Invalid coordinates, mismatched source hashes/sizes, missing origin declaration, duplicate object names, more than8objects/32points per object or unrecognized model hashes fail closed.

Only the pinned official Light HQ-SAM checkpoint is accepted. Model SHA2560f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196 matches the publisher repository LFS metadata. Revision09b02a333b37772133eff3997bdba997867374b7. Model42,526,881bytes. This command performs no download. Model loaded explicitly on CPU with weights_only=True. Source RGB is copied byte-for-byte before compositing. Binary alpha is not optical matting; hard edges, reflected source color and incorrect holes can remain.

The CLI runs each source in a separate worker with180s wall-time and Linux /proc2.5GiB RSS guard. Windows has no /proc RSS guard. Four-million-source-pixel cap; no silent source resizing. The model internally preprocesses to its own input scale. Direct Python calls to infer_file bypass the worker safeguards.

## Setup and use
Use a separate project environment. Install CPU torch/torchvision from the official PyTorch registry, then optional package dependencies from official PyPI:
```
python -m venv .prompted-env
.prompted-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.2+cpu
.prompted-env/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-prompted.txt
```
Research install used those CPU builds plus segment-anything-hq0.3 and timm1.0.31. Publisher/project links checked against official registry metadata. Windows interpreter is `.prompted-env\Scripts\python.exe`; owner-PC operation is untested. No install outside the project was made.

Download the pinned official checkpoint as a file, not a shell script, from:
https://huggingface.co/lkeab/hq-sam/resolve/09b02a333b37772133eff3997bdba997867374b7/sam_hq_vit_tiny.pth
Its original first download request returned403; a cache-busted request succeeded and the bytes matched the publisher LFS hash. Model weights are excluded from the code ZIP.

Prompt JSON example, with placeholders to replace by actual reviewed source facts:
```
{
  "source_sha256": "EXACT_SOURCE_FILE_SHA256",
  "image_size": [1125,750],
  "origin": "reviewed_source_geometry",
  "objects": [
    {"name":"notebook","box":[315,100,830,640]},
    {"name":"pen","box":[635,45,905,450],"points":[[720,160],[720,295]],"labels":[1,0]}
  ]
}
```
Do not reuse these coordinates or merely declare unreviewed geometry reviewed. A source hash/dimension match prevents applying the wrong prompts to another file; it does not prove correct geometry.
```
.prompted-env/bin/python -m renderello.cli prompted-mask --source source.jpg --prompts reviewed.json --checkpoint sam_hq_vit_tiny.pth --out new-attempt
```
Inspect every output's black/white/magenta previews, holes and parts, plus each object alpha separately. Quality and semantic completeness remain NOT_EVALUATED/NOT_VERIFIED. Do not compose cropped sources as complete products. Use a verified complete overhead source and reviewed top-down background for the existing optional opaque composition route. Do not silently switch into this manual branch during unattended generation.

## Verification and limits
- Clean core environment full suite259passed/1skipped95.50s. Nine new prompt validation/fail-closed tests.
- Separate optional-environment full suite was killed after about83% plus12tests, without a final result. Cause not verified. Do NOT call it a full optional-environment pass. Adding torch changes available backend paths in old tests.
- Actual final app CLI executed on yellow notebook+pen and complete silver scissors, including a second prompt revision. Both saved RGBA RGB arrays were checked exact against decoded original RGB. Model/source/prompt/output hashes saved.
- Matched IS-Net/HQ-SAM magenta figure inspected; notebook island/pen-hole tradeoff and scissors inner-edge issues retained, not hidden. Original high-score failures retained.
- No GPU session, paid service, web UI integration, owner-PC verification, new20-source acceptance, genuine fine-print acceptance or specialist blind comparison in this branch.

## Research synthesis and sources
1. HQ-SAM official code and Apache2.0 license: https://github.com/SysCV/sam-hq . Official paper https://arxiv.org/abs/2306.01567 . High-quality output token and multi-scale feature fusion target fine masks; the lightweight variant permits a bounded CPU test. Original paper's performance is not our product benchmark.
2. The official README points to package `segment-anything-hq` and Light HQ-SAM weights at https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth . Exact revision and SHA are recorded above. Official package https://pypi.org/project/segment-anything-hq/ .
3. WACV2026 limits study https://nickk124.github.io/projects/sfm-limits/ and paper https://arxiv.org/abs/2412.04243 . Low textural contrast and tree-like details correlate with errors across SAM variants; targeted fine-tuning does not establish a general solution. This is a warning against treating stronger prompts/high scores as acceptance.
4. VNS-SAM official repo https://github.com/guangqian-guo/VNS-SAM : explicit non-salient-feature/edge decoder direction. Researched, NOT installed/run. No superiority claimed.
5. InSPyReNet https://github.com/plemeri/InSPyReNet : high-resolution image-pyramid approach; NOT run in this branch.
6. BEN2 https://github.com/PramaLLC/BEN2/ : public base and commercial enhanced model distinct. NOT run in this branch. Do not confuse enhanced-site demos with free base results.
7. Reddit similar-color failure https://www.reddit.com/r/computervision/comments/1gois46/best_way_to_segment_objects_with_a_similarly/ : anecdotal discussion, not proof that thresholding or SAM solves same-color boundaries.
8. Reddit combined prompts https://www.reddit.com/r/computervision/comments/1lrjjov/looking_for_guidance_point_box_prompts_in_sam21/ : practitioners report boxes/points can select wrong parts or miss accessories; informed the explicit multi-object workflow, not an automatic success claim.
9. Complete silver source https://www.pexels.com/photo/silver-scissors-on-white-surface-3990146/ . Photographer Nacho Lledo, published2020. Supplemental test, not a replacement for the original20-source benchmark. Excluded craft candidate https://www.pexels.com/photo/gray-and-green-scissors-on-gray-surface-10821156/ .
10. Existing yellow source https://www.pexels.com/photo/vibrant-yellow-notebook-and-pen-flat-lay-35826069/ . Image sources/license remain in the prior evidence; no object was generated or painted into these tests.

Recommendation: keep automatic and reviewed-prompt results clearly separated. For low-contrast inputs, a visible request for source-bound boxes/points is safer than presenting a missing cover as success. Fine structures still require pixel inspection. A score gate is not a quality gate.

Prompt provenance note: all coordinates in this research were selected by the assistant after image inspection, not supplied or rated by the user or an independent human. Earlier exploratory manifests used the poorly named origin label `reviewed_human_geometry`; those are not human reviews. The published CLI uses `reviewed_source_geometry` and was rerun on both final cases, with all saved output hashes verified.
