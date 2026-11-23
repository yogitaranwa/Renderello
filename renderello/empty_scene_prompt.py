"""Empty-scene prompt construction for the background-only generation route.

The scene model receives ONLY the environment. The product is never named,
described or implied anywhere in the prompt: a prompt that describes the
product for a paragraph and then appends "no product" is contradictory, and
models follow the dominant description - that is the measured 20 Nov 2026
duplicate-product failure (a hallucinated second bottle behind the preserved
cutout).

Detailed prompts are kept LONG by design: length is not the defect,
contradiction is. All the detail describes the
empty set - surface, wall, horizon, lighting, camera, material treatment,
mood and the unobstructed placement area. Product-fidelity instructions live
in the separate product-edit prompts (whole/hybrid routes), never here.
"""

# Hard exclusion block, grounded in the archived 13 Nov 2026 empty-scene
# request (scoped/scripts/Renderello-MagicHour-empty-scene-pipeline-2146.py),
# generalized beyond one product type.
_EXCLUSIONS = (
    "Completely empty set: no product, no bottle, no packaging, no pedestal, "
    "no platform, no plinth, no riser, no props, no merchandise, no text, no "
    "logo, no label, no object silhouette, and no cast shadow from an absent "
    "object. Do not render any object. The placement area stays completely "
    "unobstructed for later photographic placement of the real item."
)

_ASPECT_WORDS = {
    "1:1": "Square commercial photograph",
    "4:5": "Vertical 4:5 commercial photograph",
    "3:4": "Vertical 3:4 commercial photograph",
    "16:9": "Wide 16:9 commercial photograph",
    "9:16": "Tall 9:16 commercial photograph",
}


class EmptyScenePrompt(str):
    """Marker type: composite_with_scene(conditioning=\"empty_guide\") only
    accepts this, so a hand-written product-bearing prompt with a \"no
    product\" tail (the 20 Nov 2026 failure shape) can never reach the
    scene model through that path."""


def build_empty_scene_prompt(template, *, aspect_key: str = "1:1",
                             horizon_frac: float = 0.60,
                             instruction: str = "") -> str:
    """Environment-only scene prompt from a template's declared set design.

    template carries scene/lighting/camera/mood text (TemplateSpec fields).
    horizon_frac is the floor-wall division as a fraction of image height,
    matched to the two-tone conditioning guide the model also receives.
    instruction is the user's environment note; it is included only after
    product-naming terms are refused, keeping the prompt non-contradictory.
    """
    if not 0.3 <= horizon_frac <= 0.8:
        raise ValueError("horizon fraction outside the usable set range")
    scene = getattr(template, "scene", "") or "seamless studio set"
    lighting = getattr(template, "lighting", "") or "soft diffused studio lighting"
    camera = getattr(template, "camera", "") or "straight-on product-photography camera height"
    mood = getattr(template, "mood", "") or "calm commercial finish"
    opener = _ASPECT_WORDS.get(aspect_key, "Commercial photograph")
    pct = round(horizon_frac * 100)
    parts = [
        f"{opener} of a completely EMPTY premium product-photography set.",
        "The supplied two-tone guide only indicates the floor-wall division; "
        "it is not an object and nothing sits on the surface.",
        f"Set: {scene}.",
        f"The floor-wall junction sits at approximately {pct} percent of "
        "image height, with the support surface continuous, level and "
        "uninterrupted across the full foreground.",
        f"Lighting: {lighting}.",
        f"Camera: {camera}.",
        f"Finish: {mood}.",
        _EXCLUSIONS,
    ]
    note = (instruction or "").strip()
    if note:
        lowered = note.lower()
        for term in ("product", "bottle", "label", "logo", "packag",
                     "jar", "tube", "box", "cream", "perfume", "spray"):
            if term in lowered:
                raise ValueError(
                    "environment instruction names the product; keep product "
                    "detail out of the empty-scene prompt")
        parts.append(f"Additional environment-only note: {note}.")
    return EmptyScenePrompt(" ".join(parts))
