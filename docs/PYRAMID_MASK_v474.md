# InSPyReNet automatic mask experiment, version 0.4.74
10 November 2026. Overall acceptance remains NO-GO.

## Result
The free automatic InSPyReNet fast model opens both silver scissors handle holes that IS-Net filled. Its inner edges look smoother in the matched source-coordinate comparison than this experiment's manually prompted HQ-SAM result. This is an observed improvement on one real supplemental source, not a universal ranking or a specialist quality rating. White/gray edge halos and low-contrast neck detail still need review.

It does NOT solve the same-color yellow cover: the cover becomes largely transparent. Seven actual CLI executions completed on the existing yellow notebook, headphones, mint shirt, open book/pen, sunglasses, numbered notebook and the new silver scissors. Agent visual inspection found:
- Silver scissors: automatic handle holes opened; edge softness and source-color fringes remain.
- Yellow notebook: cover missing. Failure.
- Headphones: center gap and thin upper bands retained; source-color edge fringe and source-cropped cable remain.
- Mint shirt: keeps neighboring decorative flowers/stems, not a product-only selection.
- Open book/pen: page largely transparent; pen retained. Source is also cropped.
- Sunglasses: lenses/background separation is unreliable. Not an optical-glass solution.
- Numbered notebook: main opaque page present; rings/edge fringe still require detailed review. No fine-print acceptance.

These seven supplemental runs do not replace the original 20-source acceptance benchmark. Original bytes/RGB and all failures are retained in evidence.

## Added code
`pyramid-mask` is an explicit optional CPU CLI. It accepts only the tested official fast checkpoint SHA256, uses package version 1.3.4, requires a supplied checkpoint, and does not download it. The model internally uses static 384x384 input and interpolates the saliency map to the original source dimensions. It does not silently resize saved source RGB. This is NOT the high-resolution base configuration and NOT optical alpha matting.

The wrapper requests `type='map'` and builds the RGBA using original decoded RGB. This matters: the upstream default RGBA path may run foreground-color estimation and change RGB. Our saved RGB arrays were checked equal to decoded source on all seven cases; this does not mean the alpha is correct or that RGB contains no source-background reflection.

A fresh temporary package config is kept inside the project, deleted afterward. No global model cache or automatic model choice. Defaults and UI are unchanged. Existing outputs are not overwritten. Four-million-source-pixel cap, per-image worker, 180 second timeout and Linux-only 2.5 GiB RSS guard. Direct `infer_file` Python calls bypass the worker protections. Windows has no /proc RSS guard. No GPU or paid service used.

## Setup and use
Use an isolated project environment with core requirements. Install CPU torch/torchvision from the official PyTorch registry first, then optional dependencies:
```
python -m venv .pyramid-env
.pyramid-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.2+cpu
.pyramid-env/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-pyramid.txt
```
The actual research environment used these CPU builds and transparent-background 1.3.4. Package/publisher links and wheel SHA were checked against official PyPI and author repositories. The initial old pip 22 resolver crashed with AssertionError before installation; upgrading pip inside the research environment to official PyPI pip 26.3 allowed installation. No global Python changes. Full installed package list and logs are in evidence.

Download the official fast checkpoint as a file:
https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth
367,520,613 bytes. Publisher config MD5: `9efdbfbcc49b79ef0f7891c83d2fd52f`, verified. Observed SHA256 and wrapper pin: `e61ed7e85763dd83a796e1d650ea78604fff524121750858c54a6b560ac58c82`.
```
.pyramid-env/bin/python -m renderello.cli pyramid-mask --source original.jpg --checkpoint ckpt_fast.pth --out new-attempt
```
Use `.pyramid-env\Scripts\python.exe` on Windows; owner-PC execution remains unverified. Inspect alpha, original source, and black/white/magenta reviews. No result is marked a quality pass.

## Base model attempt and test limits
The official 1024-base checkpoint was also downloaded and matched the publisher MD5. Two isolated base probes, yellow and scissors, exited with signal 9 after about 12 seconds without masks. Observed peak samples were about 1.84 GB RSS. The explicit 2.5 GiB guard did not fire; the kill cause is unverified. Do not call this a completed high-resolution test. No repeated base attempt or automatic downgrade was put in the CLI. The CLI only supports fast mode.

Final clean core suite: **272 passed, 1 skipped in 95.58 seconds**. Seven new fail-closed/provenance tests use mocked inference for core reproducibility. The real optional CLI runs on all seven sources are separate evidence. One earlier combined command reached its outer time limit during the core test; it had no final test result. The full core suite was then rerun on its own and completed. Do not label the optional-environment full suite passed.

Twenty-source visual acceptance, genuine fine print, eight polished templates, specialist blind comparison, owner-PC operation and UI model execution remain open.

## Research grounding
- Official paper repository: https://github.com/plemeri/InSPyReNet . Image-pyramid method and package route, MIT code.
- Primary paper: https://arxiv.org/abs/2209.09475 . ACCV publisher: https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Revisiting_Image_Pyramid_Structure_for_High_Resolution_Salient_Object_Detection_ACCV_2022_paper.html . Published benchmark claims are not our results.
- Author's package: https://github.com/plemeri/transparent-background . License: https://github.com/plemeri/transparent-background/blob/main/LICENSE . PyPI: https://pypi.org/pypi/transparent-background/json . Verified package homepage points to this author repository.
- Preprocessing/foreground-color behavior: https://github.com/plemeri/transparent-background/blob/main/transparent_background/Remover.py . Map output avoids its default RGBA color estimation.
- Official checkpoint config read at https://raw.githubusercontent.com/plemeri/transparent-background/main/transparent_background/config.yaml . Release metadata: https://api.github.com/repos/plemeri/transparent-background/releases/tags/1.2.12 . Config checksums verified; release did not supply SHA digests.
- Community caveat, not controlled evidence: https://www.reddit.com/r/StableDiffusion/comments/1o18103/best_way_to_remove_background_locally/ . A participant reports off-white boundaries breaking InSPyReNet. Supports testing same-color failures, not assuming success.
- Community debate, not benchmark: https://www.reddit.com/r/MachineLearning/comments/z2n3cl/project_background_removal_tool_based_on_our/ . A participant identifies white fringes on dark backgrounds; our black reviews retain that diagnostic.
- Scissors source: https://www.pexels.com/photo/silver-scissors-on-white-surface-3990146/ . Existing other source records remain in prior overhead evidence; this evidence includes the original bytes and SHA256s.

Recommendation: keep this as an explicit alternative, not the new default. It gives a useful automatic thin-structure example but still loses important same-color objects. Retain the reviewed-prompt fallback and inspect every part/edge; do not average away missing objects with attractive easy examples.
