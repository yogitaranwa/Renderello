# Reviewed pipeline UI, version0.4.77
10 November2026. Overall NO-GO unchanged.

The optional local `/mask-prompts` page now has an explicit **Run reviewed-mask pipeline (CPU)** button. This executes the actual pinned HQ-SAM worker on exported source-bound boxes/points, then `intent-mask` against an uploaded existing automatic result. It shows white/black/magenta reviews and downloads a ZIP. This is not a new automatic inference UI: automatic alpha must already exist. Manual object review remains required, and the intersection cannot recover pixels absent from automatic alpha.

Start in the isolated optional HQ-SAM environment, using an already downloaded pinned checkpoint:
```
python -m renderello.webui --port 8000 --out runs/webui --reviewed-checkpoint PATH_TO_sam_hq_vit_tiny.pth
```
No checkpoint path is accepted from the browser. Without the startup option the run button stays disabled with instructions. No package installation, model download, external API, GPU or paid service is triggered. The server remains bound to127.0.0.1.

Load the original, review boxes/points, check the review box and export JSON. Select a ZIP with the original automatic `source.png`, `cutout.png`, `estimated-alpha.png` and its bilateral/pyramid/semantic mask manifest all at ZIP root, then click Run. Upload cap20MB, unpacked cap40MB/20files. Flat unique filenames only, no traversal/directories/encryption; only required files are unpacked. Source hashes/RGB/dimensions/prompts/alpha consistency are checked by the existing gate. One worker per server. All attempts use new folders; failures retain FAILED.txt and no accepted result.180sHQ-SAM worker limit, Linux2.5GiBRSSguard, no WindowsRSSguard. Cross-origin browser POSTs are rejected. Not a hardened multi-user service.

Source or prompt edits invalidate results. If changed while inference runs, the stale response is not displayed. Automatic ZIP changes clear result images. A completed background output can remain in the local folder after edits or a closed page, but is not accepted for the new source. Keep the page open until the bounded request completes. No auto-retry or publication.

Actual Chromium desktop1280px and mobile390px tests loaded the complete numbered-notebook source, drew source-coordinates[375,95,835,630], exported JSON, uploaded the verified BiRefNet automatic bundle, clicked Run and completed real HQ-SAM+intersection on CPU. All three images loaded and downloaded ZIPs were recovered. No JS errors or horizontal overflow; editing prompts removed stale results/disabled run. Actual screenshot pixels inspected. Physical touch and owner-PC execution remain unverified. Two actualUI run ZIPs include intermediate masks/manifests. Final RGB/min-rule verified after download.

Full clean core suite306passed,1skipped in95.50s. Eleven new ZIP/route/config tests. One intentional duplicate-ZIP fixture produces a Python warning, not a production duplicate. No changes to automatic backend/default/template quality labels. v476edge/missing-mask caveats remain.

Broader20-source pass is separate evidence, not inferred from these two UI runs. Original20denominator includes invalid/cropped/historical sources and is not silently replaced. Full templates/scene quality,tiny print,independent blind ratings and owner-PC remain open.
