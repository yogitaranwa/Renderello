# Experimental offline CPU mask generator

Opt-in only, not the default segmentation backend or a quality acceptance.
Original publisher: https://github.com/xuebinqin/U-2-Net
Pinned architecture commit: ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29
Code (Apache-2.0): https://raw.githubusercontent.com/xuebinqin/U-2-Net/ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29/model/u2net.py
Original README-linked U2NETP weights: https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing
Torch CPU installation source: https://pytorch.org/get-started/locally/

Use a separate Python environment with CPU torch. Tested torch2.14.1+cpu from
https://download.pytorch.org/whl/cpu . No new package is required in the main app.
Download the exact original artifacts separately, then run from this code folder:

```sh
/path/to/cpu-env/bin/python -m renderello.cpu_mask \
  --source benchmark/sources/can1.jpg --code /path/to/u2net.py \
  --weights /path/to/u2netp.pth --out /path/to/fresh-mask-output
```

The command downloads nothing. It refuses code/weight SHA256 mismatches before
importing or loading them. It uses weights_only=True, one CPU thread, 320x320
inference, a separate worker, 1500MiB RSS stop and 90second timeout on Linux.
Use a fresh output directory. A stopped/failed attempt is not a finished mask.
Success requires mask-manifest.json and REVIEW_REQUIRED, never PASS.

Outputs: unmodified float raw-saliency-320.npy, source-size estimated-alpha.png,
source-RGB estimated-cutout.png, green review-preview.png, and a source/hash/timing
manifest. Normalization/resize is explicit; no component deletion or edge erosion.
Keep alpha next to its manifest to preserve estimated-model provenance when used:

```sh
python -m renderello.cli run --source benchmark/sources/can1.jpg \
  --templates clean_studio --mask /path/to/fresh-mask-output/estimated-alpha.png
```

If you upload only the PNG through the web form, it is a generic supplied mask;
web upload does not authenticate or invent the model origin. The mask sidecar is
artifact metadata, not a trusted human rating. Original GPU layers are not recovered.

## Real tests, 2026-11-09

Seven source photos: can, watch, furry teddy, boba plush, barber product set,
clear glass and sneaker. All probes completed at roughly496-537MiB observedRSS,
0.57-1.24s inference, with a separate loaded process per image. These are local
mask timings, not end-to-end generation cost or speed promises.

Agent eye review: can isolated without prior slab, but halo/faint detached text;
teddy has fringe; watch ghost fragments; boba straw partly transparent; barber
loses/fades products; glass keeps original white interior rather than true
transparency; sneaker fades. Do not promote this model automatically or infer
publish-ready quality from low memory or passing software tests. No human scores,
8-template benchmark, specialist comparison or MagicHour costs established here.


## v0.4.41 full U2NET option, still experimental

The SAME original repository/code also offers full U2NET. Use `--model u2net`
and its own original README-linked weights, not the small weights:
https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing
SHA256:10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58
Official README describes176.3MB. Same320inference and bounded worker; no downloads
performed by the command. Do not bypass a hash mismatch with a different publisher.

Matched24-source mask-only tests completed for BOTH original models. Full model
fits here too: processmaxrss about695-771MiB in24-run matrix. The actual pixels
show stronger preservation of tshirt2 black fabric/slogan, boba straw, perfume
bottle caps, and removal of can's detached text and glass2 phone/keys. However,
watch1 still loses both straps, watch2 loses straps even more completely than the
small model, barber still loses upper tools/bottles and retains unwanted content,
glass remains opaque/keeps old interior, and edge halos persist. Bigger does not
mean universally better. These are agent eye findings, not trusted human scores.
This is not the20+source8-templategenerativebenchmark, nor a bestqualityroute.

No global opacity threshold or largest-component cleanup is applied: those can
hide mistakes, remove legitimate disconnected products, and break glass or fur.
Use explicit masks only after review. Generic web alpha upload remains generic
supplied input; source/hash-bound local sidecar distinguishes model estimates.
