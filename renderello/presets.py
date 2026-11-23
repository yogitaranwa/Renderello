"""Preset library: the 8-12 reusable photoshoot templates required by the
problem statement, each with a genuinely detailed, researched prompt.

Prompt structure follows the six-part commercial-brief pattern documented in
docs/PROMPT_RESEARCH.md (product, purpose, composition, lighting,
surface/environment, constraints), with material-specific overlays so glass,
metal, fabric and packaging each get physically-correct lighting language.

Presets are data, not code paths: they sit on top of templates.py (scene
fragments) and prompts.py (fidelity guard). The studio layer assembles the
final long prompt from these parts plus the owner's free-form instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Aspect
from .templates import TEMPLATES, get_template

# ---------------------------------------------------------------------------
# Material overlays: physically accurate lighting/texture language per
# material family. Appended when the owner declares (or the intake detects)
# the product material. Grounded in the product-photography prompt research:
# glass needs controlled highlights + refraction language, metal needs
# grain/reflection relationship, matte needs tonal gradient, etc.
# ---------------------------------------------------------------------------
MATERIAL_OVERLAYS: dict[str, str] = {
    "glass": (
        "The product is transparent or translucent glass: render believable "
        "refraction and glass thickness, a controlled vertical strip highlight "
        "on one edge, subtle rim light separating the silhouette from the "
        "backdrop, and restrained internal reflections. Never frost, recolor "
        "or opacify the glass."),
    "metal": (
        "The product is metal: trace the brushed or polished grain with a long "
        "soft key so the reflection gradient runs along the grain, keep edge "
        "highlights crisp, and preserve the exact anodized or plated color. "
        "No fingerprints, no warped edges."),
    "matte_plastic": (
        "The product is matte plastic or soft-touch material: use a broad "
        "soft source that lays a smooth tonal gradient across the surface so "
        "the shape stays readable; avoid specular glare and keep the texture "
        "fine and realistic."),
    "glossy": (
        "The product is glossy or lacquered: manage reflections deliberately "
        "- one clean elongated highlight per curved face, deep saturated base "
        "color, and mirror-sharp but physically plausible surroundings "
        "reflected at reduced contrast."),
    "fabric": (
        "The product is fabric, apparel or textile: preserve weave, stitching, "
        "seams, buttons, straps and zippers exactly; light to reveal fiber "
        "texture without flattening; natural drape only, no invented folds."),
    "leather": (
        "The product is leather: emphasize grain and crease authenticity with "
        "raking soft light, keep hardware (buckles, clasps, zippers) "
        "pixel-faithful, and preserve exact dye color and finish."),
    "food_packaging": (
        "The product is food or beverage packaging: keep every label element "
        "- brand mark, variant, weight, nutrition block, barcode, MRP - "
        "bit-exact; render the wrapper's real material (foil, laminate, "
        "paperboard) and natural folds; appetizing but restrained color."),
    "skincare": (
        "The product is skincare or cosmetics packaging: airy highlights, "
        "gentle gradients on pumps/caps, clean refraction if the vessel is "
        "translucent, and perfectly legible label typography at macro "
        "fidelity."),
    "beverage_can": (
        "The product is a beverage can: manage the cylindrical highlight as "
        "one continuous strip, preserve condensation only if the scene calls "
        "for it, keep the printed artwork - every letter and graphic - "
        "exactly as on the source."),
    "ceramic": (
        "The product is ceramic or porcelain: soft broad key with gentle "
        "specular bloom on the glaze, exact glaze color, no surface "
        "repainting."),
    "electronics": (
        "The product is electronics: keep screens, ports, buttons, LEDs and "
        "printed markings exact; screens show exactly what the source shows; "
        "brushed or anodized surfaces keep their real finish."),
    "jewelry": (
        "The product is jewelry or a watch: macro-grade edge fidelity, "
        "controlled specular fire on facets, exact metal tone, legible dial "
        "or engraving, no invented stones or links."),
}

MATERIAL_OVERLAYS["glossy_plastic"] = MATERIAL_OVERLAYS["glossy"]
MATERIAL_OVERLAYS["wood"] = (
    "The product is wood: preserve the real grain direction, tone and finish "
    "exactly; light with a soft raking key that reveals texture without "
    "plastic-looking gloss; keep edges, joints and any engraving crisp; no "
    "repainted or re-colored surfaces.")

GENERIC_MATERIAL = (
    "Preserve the product's true material response - its real texture, "
    "reflectivity, translucency and finish - exactly as in the source photo.")


@dataclass(frozen=True)
class Knob:
    """One user-adjustable preset control."""
    key: str
    label: str
    kind: str                    # select | slider | toggle | text
    options: tuple[str, ...] = ()
    default: str = ""
    hint: str = ""


@dataclass(frozen=True)
class Preset:
    """A production-candidate photoshoot template with its detailed prompt."""
    key: str
    name: str
    template_key: str            # links to templates.py scene fragments
    tagline: str                 # one-line click-appeal for the picker UI
    best_for: tuple[str, ...]    # product families this preset serves
    default_materials: tuple[str, ...]
    detailed_prompt: str         # the long researched prompt body
    knobs: tuple[Knob, ...] = ()
    cover_note: str = ""         # what the hero/before-after cover shows

    @property
    def template(self):
        return get_template(self.template_key)


_K_ASPECT = Knob("aspect", "Output shape", "select",
                 ("1:1", "4:5", "3:2"), "1:1", "square, portrait or landscape")
_K_VARIANTS = Knob("variants", "Variants", "select", ("1", "2", "3", "4"), "4",
                   "alternatives to generate")
_K_SHADOW = Knob("shadow_strength", "Shadow strength", "select",
                 ("subtle", "natural", "dramatic"), "natural")
_K_PROPS = Knob("props", "Props", "toggle", (), "off",
                "allow restrained supporting props at frame edges")

PRESETS: dict[str, Preset] = {p.key: p for p in [

    Preset(
        key="clean_studio", name="Clean Studio", template_key="clean_studio",
        tagline="Marketplace-ready packshot on a seamless sweep",
        best_for=("any product", "marketplace listings", "catalog"),
        default_materials=("matte_plastic",),
        detailed_prompt=(
            "Commercial catalog photograph of the product standing on an "
            "invisible surface against a seamless soft-white studio sweep, "
            "shot as a clean ecommerce hero. Composition: front three-quarter "
            "product view, the full product visible with comfortable even "
            "margins, occupying about 70 percent of frame height, centered "
            "with generous breathing room. Lighting: a broad diffused key "
            "from the upper left and a gentle fill card from the opposite "
            "side, 5500K neutral white balance, soft shadowless wrap with a "
            "single faint contact shadow directly beneath the product. "
            "Surface and environment: seamless white-to-very-light-gray "
            "vertical gradient, no props, no set dressing. Palette and "
            "treatment: honest color, crisp micro-contrast on the product, "
            "calm neutral background. Constraints: the product's shape, "
            "proportions, colors, logos and every label character remain "
            "identical to the source; no duplicate product, no invented "
            "text, no watermark, no border, no hands."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_SHADOW),
        cover_note="paired source vs white-sweep packshot"),

    Preset(
        key="luxury_hero", name="Luxury Hero", template_key="luxury_hero",
        tagline="Dramatic dark-stage hero with rim light",
        best_for=("fragrance", "watches", "premium electronics", "spirits"),
        default_materials=("glass", "metal"),
        detailed_prompt=(
            "Premium campaign hero photograph of the product on a polished "
            "black stone slab against a dark charcoal void. Composition: "
            "slightly low hero angle with a 100mm macro product lens, tight "
            "premium crop, product slightly right of center, faint warm rim "
            "halo behind the silhouette. Lighting: one dramatic strip "
            "softbox from above and behind carving a controlled edge "
            "highlight, a warm 4300K accent kicker from the left rear, deep "
            "controlled falloff into black, and a crisp short contact shadow "
            "with a subtle mirrored reflection on the slab. Environment: "
            "near-black gradient background with no visible horizon, fine "
            "stone texture under the product. Treatment: high-contrast "
            "aspirational grade, deep blacks, specular highlights kept "
            "clean. Constraints: product geometry, finish, engravings and "
            "label typography exactly as the source; no extra products, no "
            "text, no smoke, no lens flare covering the label."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_SHADOW),
        cover_note="dark hero with edge-lit silhouette"),

    Preset(
        key="editorial_beauty", name="Editorial Beauty", template_key="editorial_beauty",
        tagline="Soft skincare-editorial pedestal scene",
        best_for=("skincare", "cosmetics", "wellness"),
        default_materials=("skincare", "glass"),
        detailed_prompt=(
            "Editorial beauty photograph of the product on a matte pastel "
            "stone pedestal in a soft beige studio. Composition: three-quarter "
            "angle at product height with a 70mm lens, generous negative "
            "space above the product, one out-of-focus botanical sprig far "
            "behind at the frame edge. Lighting: diffused north-window "
            "softbox from the left, gentle top fill, 5000K, airy highlights, "
            "a long soft shadow falling right at low contrast. Environment: "
            "soft beige-pink seamless backdrop with a gentle vignette, honed "
            "stone pedestal with subtle mineral texture. Treatment: calm "
            "editorial palette, lifted but controlled highlights, delicate "
            "contrast. Constraints: vessel, cap, pump, label typography and "
            "fill level exactly as the source; no hands, no extra bottles, "
            "no generated label text, no water droplets unless the source "
            "shows them."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_PROPS),
        cover_note="pastel pedestal beauty editorial"),

    Preset(
        key="lifestyle_kitchen", name="Lifestyle Kitchen", template_key="lifestyle_kitchen",
        tagline="Warm morning-counter lifestyle context",
        best_for=("food", "beverage", "kitchen goods", "supplements"),
        default_materials=("food_packaging", "beverage_can"),
        detailed_prompt=(
            "Lifestyle photograph of the product on a light oak countertop "
            "in a bright kitchen interior. Composition: 50mm at counter "
            "height, shallow depth of field so the product stays tack-sharp "
            "while the kitchen falls into warm bokeh, product on the left "
            "third with counter space running right. Lighting: warm morning "
            "window light from the rear-left wrapping around the product, a "
            "white bounce card filling the front, 4800K, a natural "
            "directional shadow to the right with a diffused edge. "
            "Environment: blurred bright kitchen with cream tones, soft "
            "horizontal counter line, circular bokeh in the top third, no "
            "readable packaging or brands in the background. Treatment: "
            "warm, lived-in, appetizing color with honest saturation. "
            "Constraints: package artwork, cap, seal and every printed "
            "character identical to the source; one background ingredient "
            "cue maximum, heavily defocused; no hands, no duplicate product."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_PROPS),
        cover_note="counter-top lifestyle with bokeh kitchen"),

    Preset(
        key="natural_outdoor", name="Natural Outdoor", template_key="natural_outdoor",
        tagline="Fresh open-shade outdoor setting",
        best_for=("beverages", "wellness", "eco brands", "snacks"),
        default_materials=("beverage_can", "glass"),
        detailed_prompt=(
            "Outdoor lifestyle photograph of the product on a weathered "
            "stone surface with softly blurred green foliage behind. "
            "Composition: 60mm at surface level, low horizon, the product "
            "grounded on the stone with organic texture leading toward it, "
            "background fully defocused. Lighting: open-shade daylight with "
            "dappled sun accents, 5600K, gentle sky fill lifting the shadow "
            "side, a soft dappled shadow consistent with foliage light. "
            "Environment: muted green wash backdrop with soft dappled light "
            "patches, no sky blown out, no litter or man-made clutter. "
            "Treatment: fresh organic grade, true greens, clean whites on "
            "the product. Constraints: label, cap, color and proportions "
            "identical to the source; condensation only if physically "
            "plausible for the scene; no insects, no watermarks, no extra "
            "products."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_SHADOW),
        cover_note="stone ledge with foliage bokeh"),

    Preset(
        key="minimal_tabletop", name="Minimal Tabletop", template_key="minimal_tabletop",
        tagline="Quiet architectural podium study",
        best_for=("design objects", "cosmetics", "home goods"),
        default_materials=("ceramic", "matte_plastic"),
        detailed_prompt=(
            "Minimalist tabletop photograph of the product on a single matte "
            "neutral podium against a plain light-gray sweep. Composition: "
            "85mm straight-on at product center, symmetrical framing, the "
            "podium edge low in frame, large areas of quiet negative space. "
            "Lighting: one large overhead softbox with minimal fill, 5200K, "
            "sculpted but simple, a single soft elliptical shadow under the "
            "podium edge. Environment: uniform light gray with a very subtle "
            "floor-wall seam low in frame, no props, no texture noise. "
            "Treatment: architectural, desaturated-neutral palette, precise "
            "edges. Constraints: exact product silhouette, finish and "
            "markings preserved; no color cast on the product; no extra "
            "objects; no text."),
        knobs=(_K_ASPECT, _K_VARIANTS),
        cover_note="single podium minimal study"),

    Preset(
        key="seasonal_holiday", name="Seasonal Holiday", template_key="seasonal_holiday",
        tagline="Festive warm-bokeh campaign look",
        best_for=("gifting", "food & beverage", "limited editions"),
        default_materials=("glossy", "food_packaging"),
        detailed_prompt=(
            "Festive campaign photograph of the product on a holiday "
            "tabletop with restrained seasonal accents. Composition: 70mm "
            "slightly elevated 15 degrees, product as the undisputed hero "
            "center-frame, one pine sprig and warm bokeh lights confined to "
            "the far background edges only. Lighting: warm tungsten key at "
            "3800K with a cool rim light for separation, gentle sparkle "
            "highlights in the bokeh, a warm soft shadow with a subtle "
            "festive glow at the base. Environment: deep warm amber gradient "
            "backdrop with distant gold bokeh, matte tabletop with fine "
            "texture. Treatment: festive but premium - rich ambers, deep "
            "greens, never cluttered, never kitsch. Constraints: product "
            "artwork, ribbons and labels identical to the source; no "
            "ornaments touching or occluding the product; no generated "
            "greeting text; no snow on the product unless requested."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_PROPS),
        cover_note="warm bokeh festive hero"),

    Preset(
        key="social_ad", name="Social Ad", template_key="social_ad",
        tagline="Bold graphic scroll-stopper with copy space",
        best_for=("social campaigns", "launches", "ads"),
        default_materials=("glossy",),
        detailed_prompt=(
            "Vertical 4:5 social advertisement photograph of the product "
            "against a bold single-color backdrop with a graphic diagonal "
            "light band. Composition: 50mm with a dynamic slight tilt, the "
            "product upper-center and filling enough of the frame to stay "
            "legible on a phone screen, deliberate open copy space at the "
            "bottom-left for later headline placement. Lighting: hard "
            "directional key producing a defined graphic drop shadow offset "
            "down-right, a colored gel accent from the rear, high contrast "
            "with a crisp product silhouette. Environment: saturated solid "
            "backdrop with a diagonal lighter band, product floating "
            "slightly above a implied surface. Treatment: energetic modern "
            "ad grade, punchy but believable color. Constraints: product "
            "geometry, logos and label text exactly as the source; no "
            "generated words anywhere in frame; no extra products; nothing "
            "in the reserved copy space; no watermark."),
        knobs=(_K_VARIANTS, _K_SHADOW,
               Knob("backdrop_color", "Backdrop color", "text", (), "brand red",
                    "a solid color or hex, e.g. #C8102E")),
        cover_note="bold diagonal-band ad layout"),

    Preset(
        key="reflective_surface", name="Reflective Mirror", template_key="reflective_surface",
        tagline="Sleek black-acrylic mirror reflection",
        best_for=("electronics", "fragrance", "watches", "tech"),
        default_materials=("metal", "glass", "glossy"),
        detailed_prompt=(
            "Sleek product photograph on glossy black acrylic with a clear "
            "mirror reflection below the product. Composition: 90mm at "
            "surface level, product centered with its reflection extending "
            "naturally below, dark gradient studio behind with no visible "
            "horizon. Lighting: two strip softboxes left and right creating "
            "symmetric edge highlights, 5000K, the reflection fading "
            "gradually with distance, minimal ambient shadow. Environment: "
            "dark neutral gradient, the acrylic's fine reflection haze kept "
            "physically plausible. Treatment: polished tech grade, deep "
            "blacks, controlled speculars. Constraints: the reflection must "
            "mirror the real product geometry - never an invented bottom "
            "side; product markings, ports and labels exactly as the source; "
            "no duplicated product beside the reflection; no text."),
        knobs=(_K_ASPECT, _K_VARIANTS),
        cover_note="black acrylic mirror product study"),

    Preset(
        key="flat_lay", name="Flat Lay", template_key="flat_lay",
        tagline="Top-down editorial arrangement",
        best_for=("apparel", "accessories", "stationery", "multi-product sets"),
        default_materials=("fabric", "leather"),
        detailed_prompt=(
            "Top-down flat lay photograph of the product on textured linen "
            "viewed directly from above. Composition: camera at exactly 90 "
            "degrees overhead, product centered with even margins, at most "
            "two minimal side props resting at the corners, everything "
            "aligned to a quiet grid. Lighting: a large overhead diffused "
            "source, perfectly even with no directional bias, 5500K, only a "
            "very soft ambient-occlusion ring under the product. "
            "Environment: uniform light textile texture, natural weave "
            "visible at close range. Treatment: tidy graphic editorial "
            "grade, true fabric color, crisp edges. Constraints: fabric "
            "weave, stitching, hardware and print exactly as the source; no "
            "folds invented across logos; no extra items beyond the corner "
            "props; no text."),
        knobs=(_K_ASPECT, _K_VARIANTS, _K_PROPS),
        cover_note="top-down linen arrangement"),

    Preset(
        key="macro_detail", name="Macro Detail", template_key="minimal_tabletop",
        tagline="Raking-light texture close-up",
        best_for=("craft details", "materials", "quality proof"),
        default_materials=("leather", "metal", "fabric"),
        detailed_prompt=(
            "Macro commercial photograph focusing on the product's defining "
            "material or construction feature - its weave, grain, engraving, "
            "stitching or mechanism. Composition: tight macro crop on the "
            "feature with smooth background falloff, the rest of the product "
            "dissolving into soft blur. Lighting: raking soft side light "
            "revealing surface texture, sharp detail across the important "
            "feature, neutral color reproduction. Environment: minimal "
            "out-of-focus neutral backdrop. Treatment: honest macro texture, "
            "no oversharpening halo. Constraints: every texture element must "
            "match the real product - no altered stitching, no changed "
            "hardware, no invented engraving, no text, no dust or scratches "
            "added."),
        knobs=(_K_VARIANTS, _K_SHADOW),
        cover_note="macro texture detail"),

    Preset(
        key="hero_negative_space", name="Hero + Copy Space", template_key="luxury_hero",
        tagline="Website hero with room for your headline",
        best_for=("landing pages", "banners", "campaigns"),
        default_materials=("glossy",),
        detailed_prompt=(
            "Wide website-hero photograph of the product positioned on the "
            "right third of the frame on a minimal stone surface. "
            "Composition: landscape banner framing, the product anchored "
            "right-of-center with a clean open area on the left reserved for "
            "later headline and button placement - absolutely no objects or "
            "texture clutter in the reserved area. Lighting: soft "
            "directional light from the upper left, a realistic contact "
            "shadow, refined falloff toward the left. Environment: minimal "
            "surface with a quiet gradient wall, subtle vignette pulling "
            "attention to the product. Treatment: refined commercial "
            "campaign grade. Constraints: product exactly as the source; no "
            "generated text or placeholder glyphs anywhere; no duplicate "
            "product; the left half stays clean enough for white headline "
            "type."),
        knobs=(_K_VARIANTS,),
        cover_note="banner with clean headline zone"),
]}

assert len(PRESETS) >= 8, "problem statement requires at least 8 templates"


def get_preset(key: str) -> Preset:
    try:
        return PRESETS[key]
    except KeyError:
        raise KeyError(
            f"unknown preset '{key}'. available: {', '.join(sorted(PRESETS))}"
        ) from None


def material_overlay(material: str | None) -> str:
    """Resolve a declared/detected material to its overlay text."""
    if not material:
        return GENERIC_MATERIAL
    return MATERIAL_OVERLAYS.get(material.lower().strip(), GENERIC_MATERIAL)


def build_detailed_prompt(preset: Preset, *, product_description: str = "",
                          material: str | None = None,
                          instruction: str = "",
                          reference_style: str = "",
                          knob_values: dict[str, str] | None = None) -> str:
    """Assemble the full long prompt for one generation.

    Order is deliberate: product identity and fidelity constraints lead
    (editor models attend to the front), then the researched scene body,
    then material physics, reference style, knobs and the owner's free-form
    instruction. Nothing is length-capped here; longprompt.plan_prompt
    handles provider-specific feeding.
    """
    parts: list[str] = []
    if product_description:
        parts.append(f"The product: {product_description.strip()}.")
    parts.append(preset.detailed_prompt.strip())
    parts.append(material_overlay(material))
    if reference_style:
        parts.append(f"Match this reference style: {reference_style.strip()}.")
    kv = knob_values or {}
    color = (kv.get("backdrop_color") or "").strip()
    if color:
        parts.append(f"The backdrop color is {color}.")
    if kv.get("shadow_strength") == "dramatic":
        parts.append("Push the shadow deeper and more directional.")
    elif kv.get("shadow_strength") == "subtle":
        parts.append("Keep the shadow barely-there and soft.")
    if instruction.strip():
        parts.append(f"Owner instruction (scene and environment only, never "
                     f"the product): {instruction.strip()}.")
    parts.append("Photorealistic commercial product photography, high "
                 "resolution, sharp focus on the product.")
    return " ".join(parts)


def presets_for_ui() -> list[dict]:
    """JSON-able preset cards for the website / notebook pickers."""
    out = []
    for p in PRESETS.values():
        t = p.template
        out.append({
            "key": p.key, "name": p.name, "tagline": p.tagline,
            "category": t.category, "aspect": t.aspect.value,
            "best_for": list(p.best_for),
            "materials": list(p.default_materials),
            "knobs": [{"key": k.key, "label": k.label, "kind": k.kind,
                       "options": list(k.options), "default": k.default,
                       "hint": k.hint} for k in p.knobs],
            "cover_note": p.cover_note,
        })
    return out
