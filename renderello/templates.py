"""Shoot template library. Draft set per the research plan Part D; the final set
is chosen from evaluated quality (Phase 0/1 gate), so templates are data, not code
paths - add or drop entries here without touching the pipeline.

Fragment discipline: scene/lighting/camera/mood describe ONLY the environment.
Nothing here may restyle the product itself; prompts.py adds the hard fidelity
guard around these fragments.
"""
from __future__ import annotations

from .schemas import Aspect, TemplateSpec

TEMPLATES: dict[str, TemplateSpec] = {t.key: t for t in [
    TemplateSpec(
        key="clean_studio", name="Clean Studio / White Background",
        category="catalog",
        scene="seamless white studio backdrop, product centered on an invisible surface",
        lighting="large soft key light front-left, fill card right, even shadowless wrap, 5500K neutral",
        camera="85mm product lens, eye-level, product occupying 70% of frame height",
        mood="clinical, honest, marketplace-ready",
        aspect=Aspect.SQUARE,
        background_hint="neutral white to very light gray vertical gradient",
        shadow="soft faint contact shadow directly beneath product",
    ),
    TemplateSpec(
        key="luxury_hero", name="Luxury Hero",
        category="hero",
        scene="polished black stone slab, dark charcoal void background, subtle rim glow",
        lighting="single dramatic strip softbox from above-behind, warm 4300K accent kicker, deep controlled falloff",
        camera="100mm macro product lens, slightly low hero angle, tight premium crop",
        mood="premium, aspirational, high-contrast",
        aspect=Aspect.PORTRAIT,
        background_hint="near-black gradient with faint warm rim halo behind product",
        shadow="crisp short contact shadow, slight reflection on slab",
    ),
    TemplateSpec(
        key="editorial_beauty", name="Editorial Beauty / Skincare",
        category="beauty",
        scene="matte pastel stone pedestal, soft beige studio, one out-of-focus botanical sprig far behind",
        lighting="diffused north-window softbox left, gentle top fill, 5000K, airy highlights",
        camera="70mm, three-quarter angle at product height, generous negative space above",
        mood="clean, editorial, calm skincare aesthetic",
        aspect=Aspect.PORTRAIT,
        background_hint="soft beige-pink seamless with gentle vignette",
        shadow="long soft shadow falling right, low contrast",
    ),
    TemplateSpec(
        key="lifestyle_kitchen", name="Lifestyle Kitchen",
        category="lifestyle",
        scene="light oak countertop, blurred bright kitchen interior behind, morning atmosphere",
        lighting="warm window light from rear-left, white bounce card front, 4800K",
        camera="50mm at counter height, shallow depth of field, background bokeh",
        mood="warm, lived-in, appetizing",
        aspect=Aspect.LANDSCAPE,
        background_hint="warm cream tones, soft horizontal counter line, bokeh circles top third",
        shadow="natural directional shadow to the right, diffused edge",
    ),
    TemplateSpec(
        key="natural_outdoor", name="Natural Outdoor",
        category="lifestyle",
        scene="weathered stone or wood surface outdoors, softly blurred green foliage background",
        lighting="open-shade daylight with dappled sun accents, 5600K, gentle sky fill",
        camera="60mm at surface level, low horizon, background defocused",
        mood="fresh, organic, eco",
        aspect=Aspect.LANDSCAPE,
        background_hint="muted green wash with soft dappled light patches",
        shadow="soft dappled shadow consistent with foliage light",
    ),
    TemplateSpec(
        key="minimal_tabletop", name="Minimalist Tabletop",
        category="catalog",
        scene="single matte neutral podium on a plain light-gray sweep, no props",
        lighting="one large overhead softbox, minimal fill, 5200K, sculpted but simple",
        camera="85mm, straight-on at product center, symmetrical framing",
        mood="minimal, architectural, quiet",
        aspect=Aspect.SQUARE,
        background_hint="uniform light gray, very subtle floor-wall seam low in frame",
        shadow="single soft ellipse under podium edge",
    ),
    TemplateSpec(
        key="seasonal_holiday", name="Seasonal / Holiday",
        category="campaign",
        scene="festive tabletop with restrained seasonal accents (pine sprig, warm bokeh lights far behind)",
        lighting="warm tungsten key 3800K with cool rim separation, gentle sparkle highlights",
        camera="70mm, slightly elevated 15 degrees, product hero with accents at frame edges only",
        mood="festive but premium, never cluttered",
        aspect=Aspect.SQUARE,
        background_hint="deep warm amber gradient with distant gold bokeh",
        shadow="warm soft shadow, subtle festive glow at base",
    ),
    TemplateSpec(
        key="social_ad", name="Social Ad",
        category="campaign",
        scene="bold single-color backdrop with graphic diagonal light band, product floating slightly above surface",
        lighting="hard directional key for a defined graphic shadow, colored gel accent from rear, high contrast",
        camera="50mm, dynamic slight tilt, product upper-center leaving copy space bottom-left",
        mood="energetic, thumb-stopping, modern ad",
        aspect=Aspect.PORTRAIT,
        background_hint="saturated solid color with diagonal lighter band",
        shadow="hard-edged graphic drop shadow offset down-right",
    ),
    TemplateSpec(
        key="reflective_surface", name="Reflective Surface / Mirror",
        category="hero",
        scene="glossy reflective black acrylic surface, dark gradient studio behind",
        lighting="two strip softboxes left/right for symmetric edge highlights, 5000K",
        camera="90mm at surface level, product with clear mirror reflection below",
        mood="sleek, tech, polished",
        aspect=Aspect.SQUARE,
        background_hint="dark neutral gradient, no visible horizon",
        shadow="reflection fades with distance, minimal ambient shadow",
    ),
    TemplateSpec(
        key="flat_lay", name="Flat Lay / Top-Down",
        category="catalog",
        scene="textured linen or paper surface viewed directly from above, minimal side props at corners",
        lighting="large overhead diffused source, perfectly even, no directional bias, 5500K",
        camera="top-down 90 degrees, product centered with even margins",
        mood="tidy, graphic, editorial catalog",
        aspect=Aspect.SQUARE,
        background_hint="uniform light textile texture",
        shadow="very soft ambient occlusion ring only",
    ),
]}

assert len(TEMPLATES) >= 8, "problem statement requires at least 8 templates"


def get_template(key: str) -> TemplateSpec:
    try:
        return TEMPLATES[key]
    except KeyError:
        raise KeyError(f"unknown template '{key}'. available: {', '.join(sorted(TEMPLATES))}") from None


def all_templates() -> list[TemplateSpec]:
    return list(TEMPLATES.values())
