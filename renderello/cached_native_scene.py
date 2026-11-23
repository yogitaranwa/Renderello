"""Repeatable native product paste onto a saved scene, with authored shadow.

No model calls or fit-to-canvas resizing. Matte is interpreted as ordinary
straight-alpha coverage in sRGB code values, matching Pillow source-over.
Shadow darkening uses linear light; this is not calibrated physical relighting.
"""
import numpy as np
from PIL import Image
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .checked_research.contact_camera import linear, srgb


def compose_cached_native(source, alpha, scene, *, translation, shadow=None):
    native_rgb(source); native_rgb(scene); native_alpha(alpha, source.shape[:2])
    sh, sw = alpha.shape
    dh, dw = scene.shape[:2]
    if sh * sw > 1_000_000 or dh * dw > 2_000_000:
        raise ValueError('native1MPproduct/2MPscene limits; no automatic resize')
    if not isinstance(translation, (tuple, list)) or len(translation) != 2 or any(type(v) is not int or v < 0 for v in translation):
        raise ValueError('nonnegative integer native translation required')
    x, y = translation
    if x + sw > dw or y + sh > dh:
        raise ValueError('whole native product must fit; no silent clipping or resize')
    if not np.any(alpha):
        raise ValueError('empty product alpha')
    if shadow is None:
        dark = scene.copy()
    else:
        if not isinstance(shadow, np.ndarray) or shadow.dtype.kind != 'f' or shadow.shape != (dh, dw) or not np.isfinite(shadow).all() or np.any((shadow < 0) | (shadow > .5)):
            raise ValueError('finite native shadow attenuation0..0.5required')
        if np.any(shadow[y:y+sh, x:x+sw][alpha >= 128]):
            raise ValueError('authored shadow may not cross hard product support')
        dark = srgb(linear(scene) * (1-shadow[..., None]))
        dark[shadow == 0] = scene[shadow == 0]
    canvas = Image.fromarray(dark).convert('RGBA')
    rgba = Image.fromarray(np.dstack((source, alpha)), 'RGBA')
    canvas.alpha_composite(rgba, (x, y))
    output = np.array(canvas.convert('RGB'))
    interior = alpha == 255
    if not np.array_equal(output[y:y+sh, x:x+sw][interior], source[interior]):
        raise RuntimeError('native opaque product RGB invariant failed')
    return output, dict(schema='cached-native-scene-composite.v1', source_sha256=digest(source),
        alpha_sha256=digest(alpha), scene_sha256=digest(scene), result_sha256=digest(output),
        shadow_sha256=digest(shadow) if shadow is not None else None,
        translation=[x,y], source_native_size=[sw,sh], output_size=[dw,dh],
        source_resampled=False, source_rgb_repainted=False,
        opaque_source_rgb_exact_pixels=int(interior.sum()),
        alpha_space='straight coverage, sRGB code-value source-over',
        shadow_space='linear RGB attenuation, authored not recovered',
        extra_api_calls=0, physical_relighting=False, photo_accepted=False,
        limitations=['Fractional alpha blends source and scene; exactness applies only to fully opaque pixels.',
                     'Matte quality, source lighting, material optics and scene compatibility are not established.',
                     'Authored contact and shadows are not measured scene geometry.'])
