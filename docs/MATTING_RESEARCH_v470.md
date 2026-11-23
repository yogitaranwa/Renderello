# Matting research and opt-in implementation
10 November 2026. Research build 0.4.70. Overall product acceptance: NO-GO.

The most useful tested addition is foreground-color recovery, separate from alpha refinement. CF/LKM plus multi-level foreground estimation reduced cyan contamination around one overhead notebook. It is not a general mask fix: bad semantic seeds still lose the yellow cover or retain orange inside headphones. Reviewed headphone seeds improve the hole but leave haze, and the source cable remains cropped.

## Ranked methods
| Priority | Method | Evidence and decision |
| --- | --- | --- |
| 1 | CF/LKM alpha plus foreground recovery | Implemented and CPU-tested. CF notebook: 121,404 protected core pixels, RGB max error 0; 15,005 changed edge RGB pixels. Ring ghosts remain. Keep opt-in. |
| 2 | Reviewed trimap corrections | Implemented input; tried on headphones. Removes much of inner orange, but haze remains. Per-image human work, not automatic segmentation success. |
| 3 | Guided alpha filter | Implemented from RGB guided-filter equations. Fast, no new dependency. Notebook cyan persists because source edge colors remain contaminated. Not a sufficient fix. |
| 4 | BEN2 or InSPyReNet mask alternative | Official code/license researched; NOT installed or run. BEN2 public base differs from its enhanced commercial model. Free GPU comparison can test semantic failures without inventing a win. |
| 5 | Morphological erosion / color-seeded GrabCut | Previously tried. Erosion loses detail without reliably removing cyan; seeded GrabCut jagged. Not promoted. |

## Use
Existing defaults and UI are unchanged. This is an explicit CLI research branch.

Optional dependency, from official PyPI only, inside the Renderello project environment:
```
python -m venv .matting-env
.matting-env/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-matting.txt
```
Windows uses `.matting-env\Scripts\python.exe`; the owner-PC installation is not tested. The Linux research environment already had the normal app dependencies. Install ordinary app requirements into a fresh environment too. Never run shell scripts fetched from URLs.

```
python -m renderello.cli matte-refine --source original.png --alpha estimated-alpha.png --method cf --estimate-edge-colors --out matting-attempt
python -m renderello.cli matte-compose --bundle matting-attempt --background reviewed-overhead.png --source-pose overhead --source-material opaque --source-framing complete --out composed-attempt
```
Use the interpreter where optional requirements are installed for CF/LKM or foreground recovery. `--method guided` without edge recovery uses normal app dependencies only. `--trimap reviewed-trimap.png` accepts source-sized grayscale 0=background,128=unknown,255=foreground. Supplied trimaps are reviewer inputs, not verified ground truth. Automatic seeds from alpha can lock semantic mistakes. The pose/material/framing declarations require actual review, not automatic inference. A background must really be top-down too.

Inspect black, white, magenta previews, trimap, changed-RGB map and all holes before compositing. New output directories only; originals and old attempts are retained. A one-million-source-pixel cap and a 200,000-unknown-pixel cap refuse oversized solves without resizing. CLI refinement has a 90-second subprocess timeout and a 1,500 MiB RSS check on Linux `/proc`; RSS enforcement is NOT available on Windows. No network requests or model downloads occur during refinement. Manual direct Python calls to `refine_file` bypass the process guards.

`matte-compose` verifies saved bundle file hashes and consumes recovered foreground RGB plus alpha together, preserving original-source hashes in its output manifest. Do NOT pass just the refined alpha into the old original-RGB route and assume foreground recovery survived. This route rejects declarations other than overhead/opaque/complete. It cannot independently prove those declarations. Hashes provide integrity, not semantic correctness or authenticated provenance. Placement can resample product pixels once; the shadow is an approximate 2D contact effect, not recovered physical lighting.

Protected-core RGB error zero refers to pre-placement pixels marked definite foreground by that trimap. It does not establish correct masks, exact edge colors, readable fine print after scaling, completeness or natural contact. Edge-color recovery is explicitly estimated, never whole-product source-exact. Transparent glass, pose changes, cropped objects and relighting remain outside this opaque route.

## Verification
- Full default CPU suite: 246 passed, 1 skipped, 91.99 s.
- Optional real PyMatting CF and LKM synthetic integration executed: fixed foreground/background seeds, protected core exact, no RGB changes outside unknown region.
- Final real notebook `matte-refine` and hash-checked `matte-compose` CLI chain executed and visually inspected on stone (512x640). No genuine fine-print acceptance case here.
- Notebook raw/refined matched comparison: same source, background, placement and 2x display crop; reduced cyan, residual ring ghosts.
- Yellow/headphones automatic-seed failures and reviewed-seed headphone limitation retained as negative evidence.
- Owner PC, web UI control for this stage, 20-source acceptance, 8 polished templates and specialist blind comparison are NOT verified by this update.
- No paid service, Magic Hour run or GPU session in this branch.

## Sources and interpretation
1. Official PyMatting code and MIT license: https://github.com/pymatting/pymatting/ . Alpha solvers and foreground estimators are distinct stages. https://pymatting.github.io/alpha.html and https://pymatting.github.io/foreground.html .
2. Official registry and install docs: https://pypi.org/project/PyMatting/ and https://pymatting.github.io/installation.html . Research install pinned to1.1.16, publisher/project links checked against official GitHub. Registry metadata retained with evidence.
3. Primary paper, Fast Multi-Level Foreground Estimation: https://arxiv.org/abs/2006.14970 . Foreground RGB recovery is a separate inverse-compositing problem; it cannot turn wrong semantic masks into correct ones.
4. PyMatting JOSS paper: https://joss.theoj.org/papers/10.21105/joss.02481 . Library context, not a benchmark proving our product quality.
5. Guided Image Filtering, He/Sun/Tang: https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf . Local implementation follows RGB equations14-16. Not foreground reconstruction.
6. Official BEN2 base implementation: https://github.com/PramaLLC/BEN2/ . MIT public base, foreground refinement option; enhanced commercial model is distinct. Not tested here.
7. Official InSPyReNet: https://github.com/plemeri/InSPyReNet . MIT high-resolution salient-object method; NOT tested here.
8. Official BiRefNet: https://github.com/ZhengPeng7/BiRefNet/ . Prior method and pending overhead GPU notebook; this research branch did not execute that comparison.
9. Community evidence: https://www.reddit.com/r/photoshop/comments/1am5ga7/uyork0511m_was_tasked_to_take_product_pictures/ discusses environment reflections. Reflections belong to source RGB, not just alpha. Manual reflection changes would violate exact-RGB claims.
10. Community evidence: https://www.reddit.com/r/photoshop/comments/1javk7t/tried_the_tutorials_on_defringing_and_using_the/ discusses contaminated edge colors and defringing. Anecdotal leads only, no benchmark or verified superiority.
11. Actual test photographs: https://www.pexels.com/photo/notebook-1314540/ , https://www.pexels.com/photo/vibrant-yellow-notebook-and-pen-flat-lay-35826069/ , https://www.pexels.com/photo/black-headphone-on-orange-background-20385205/ . License: https://www.pexels.com/license/ . Notebook uses an explicitly recorded crop, not invented product content.

All listed substantive sources were fetched during this research. Remaining uncertainty is quality across diverse sources, better semantic segmentation, small ring/cable detail, real contact/shadows, owner-device operation and specialist evaluation. The research narrows the next experiment; it does not satisfy the acceptance contract.
