# Prompt research (2026-11-20)

## Structure used in `presets.py`
Every preset prompt follows a six-part structure, from product-photography
prompt research on FLUX-class editor models
(https://ai-image-toolkit-smoky.vercel.app/blog/flux-product-photography-prompts):
1. Shot type and framing (hero, flat lay, macro...)
2. Scene and surfaces (what the product sits on/in)
3. Lighting (key/fill/rim, direction, quality, color temperature)
4. Materials and physics (how light interacts with glass/metal/fabric...)
5. Camera and lens language (85mm product lens, f/8, focus stacking...)
6. Output constraints (photorealistic, sharp product focus)

## Material overlays
`presets.material_overlay` maps declared/detected materials (matte/glossy
plastic, glass, metal, fabric, ceramic, wood, leather) to a physics paragraph
- e.g. glass gets refraction/caustics language, metal gets anisotropic
highlight control. Wrong or generic material language is a visible failure
mode in editor models, so overlays are explicit per material.

## Long prompts without truncation
CLIP-based encoders truncate at 77 tokens. `longprompt.py` chunks the full
prompt and mean-pools embeddings (diffusers-native), and
`long_prompt_sdxl.py` (Nov 14 tree) does dual-CLIP encoding via Compel with
`truncate_long_prompts=False`. Magic Hour's editor takes natural-language
prompts directly, so the full researched text is passed untruncated there.

## Reference images
`multinput.extract_reference_style` measures palette, warmth, brightness,
saturation and contrast, and turns them into a prompt fragment
("Style reference: ..."). References steer the scene only - they are never
composited, and the prompt says so explicitly.

## Live findings (round 1, 2026-11-20, docs/evidence/mh-live-round1/)
Detailed scene prompts produce good ENVIRONMENTS on both tested editor
models. Product fidelity is the failure point: all four live outputs carried
critical product alterations (cap/pump/proportion changes or outright product
replacement). Prompt detail does not fix that; treat every whole-image edit
as eye-review-required until the fidelity gate is wired and passing.
