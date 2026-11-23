# Explicit image intake and edge-case coverage
11 November 2026. Optional command, not a change to existing generation defaults.

`python -m renderello.image_intake --source original.jpg --out new-intake`

Embedded ICC requires `--allow-colour-conversion`, which uses the supplied profile to convert to sRGB and records that pixels may change. Unprofiled CMYK is refused, not guessed. RGB without a profile is explicitly treated as sRGB with a warning; this is not calibrated colour verification. Alpha, including zero/partial alpha, is retained. EXIF orientation is normalized on a derivative only. EXIF/GPS/comments are not exported. Original files remain unchanged. New output folders only.

Inputs have explicit byte, pixel, side-length and aspect budgets. Animated/multipage and unsupported formats/modes are refused. 16-bit/float images need a separate tone-mapping path and are refused rather than silently truncated. A special PNG header check catches RGB16 images that Pillow exposes as ordinary RGB after quantization. Corrupt/truncated inputs fail. No external parser, package download or model call occurs in this command.

Output writing stages and round-trip verifies PNG pixels, reserves a new destination and records a completion JSON. It is not a transactional multi-file database write; a mid-move failure retains an INTAKE_INCOMPLETE marker. A concurrent caller cannot overwrite an existing destination. An empty staging directory is removed; nonempty failure evidence is preserved.

Tests: 47 intake tests passed; full core suite 523 passed, one skipped, one existing intentional duplicate-ZIP warning. These tests are about exact decoding, policy, orientation, transparency and non-overwrite properties, not visual publication quality. Initial verify-after-metadata ordering failed 11 tests and was corrected by verifying a fresh stream before metadata access. No test was weakened to make the implementation pass.

27 saved JPG inputs were decoded and reviewed side-by-side, including the known illustrated shirt source; this is not a 27-real-photo quality benchmark. All originals retained exact file hashes. Optics, semantic masks, print legibility, ground contact, crop completion, reflective lighting, fur boundaries and scene quality remain unverified by intake. A successful intake never returns an automatic publication pass. Existing generation entry points do not yet use this module; promotion requires route-specific integration and visual checks.

Sources checked:
- https://pillow.readthedocs.io/en/stable/reference/Image.html
- https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
- https://pillow.readthedocs.io/en/stable/reference/ImageCms.html
