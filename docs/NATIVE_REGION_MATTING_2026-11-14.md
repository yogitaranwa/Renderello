# Native region matting integration
14 November 2026, 09:18 IST. Opt-in research code; no default backend changed.

`matting.refine_native_region` now allows a small reviewed region of an up-to-8MP native source to use the existing closed-form/LKM/guided alpha solver and optional foreground-colour recovery. The old whole-image solver refuses sources over one million pixels. A 1024x1536 garment therefore previously needed a separate manual crop experiment. The new function solves only the supplied native region, applies a border taper back to the original prior, and returns full-size arrays without resizing or passing the whole source to a float solver.

The caller supplies exact decoded-array source/alpha digests, integer source-space ROI, review note, explicit opaque/opaque-fibres declaration and optional matching trimap. Digests use `candidate_diagnostics.digest`; these are not encoded-file SHA256 values. ROI size retains the existing one-million-pixel solver cap and unknown-band cap. No automatic download, network, model call, service signup or default routing change exists. This array function has allocation/size checks, not an enforced wall-clock worker timeout.

Only trimap-unknown pixels can change, with the rectangle border exactly retained. Outside-ROI RGB and alpha remain identical. Definite foreground/background seeds remain identical. Optional edge-colour estimation changes boundary RGB and is explicitly not global source-exact fidelity. A feather can reduce a rectangular seam but also retain errors in the old prior near the ROI border. The trimap/ROI are reviewer hypotheses, not optical or semantic truth.

## Saved real-source replay

Source is the previously saved native 1024x1536 photographed blue garment, original file SHA256 b8a58a9ddb89051654642a3471fade75747fa78a4bfeb8a77a285eff332e8bb2. The prior remains the earlier manually supervised GrabCut result. ROI (720,420,840,580) is 120x160 native pixels and covers the folded right-sleeve lip. The existing two-pixel derived trimap is reused, not presented as a new independent annotation.

Full-source integration completed in 1.42 seconds with no resampling: 936 alpha pixels and 1002 boundary-colour pixels changed inside the ROI. All other source pixels/alpha and definite trimap seeds are unchanged. The 11,511-pixel definite crop foreground has zero RGB error. Source/prior files were not edited. Runtime is one local measurement, not a deployment or median-latency benchmark.

Inspected the actual 2x-nearest native-region board on white/dark/magenta and the full cutout. Compared with the binary prior, the edge steps around the sleeve lip are smoother. Most of that benefit was already present in the earlier alpha-only crop. Foreground-colour recovery makes a small additional boundary difference, not a broad photorealism gain. Blue-gray fringe and the ambiguity between dark fabric and background remain. The hanging view, wrinkles, neckline occlusion and other mask edges are unchanged. No hidden fabric, better camera, scene relighting or physical contact is inferred. This is not a polished template or full-garment acceptance.

Tests cover exact outside/border preservation, definite seeds, optional colour estimation, native source larger than the whole-image solver cap, hash binding, bad coordinates, material/scope/feather checks. Seventy-nine matting-focused tests passed in 0.52 seconds. No full-project regression or quality score is claimed.
