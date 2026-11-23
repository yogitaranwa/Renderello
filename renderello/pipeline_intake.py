"""Shared guarded image decode for generation. No photo-quality acceptance.

Colour conversion is explicit. Pixel identity after normalization is a different
claim from equality with an original file's encoded RGB values.
"""
import hashlib
from .image_intake import inspect_image, IntakePolicy


def load_generation_image(path, *, allow_colour_conversion=False, use_embedded_alpha=False):
    result = inspect_image(path, IntakePolicy(allow_colour_conversion=allow_colour_conversion))
    try:
        image = result.image.convert('RGB')
        image.info.clear()
        record = dict(result.record)
        record['normalized_rgb_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
        record['normalized_rgb_size'] = list(image.size)
        record['pixel_comparison_basis'] = 'normalized RGB derivative, not original encoded colour values'
        if type(use_embedded_alpha) is not bool:
            raise ValueError('embedded alpha choice must be boolean')
        if use_embedded_alpha and result.image.mode != 'RGBA':
            raise ValueError('use embedded alpha requires a source with alpha/transparency')
        alpha = result.image.getchannel('A') if use_embedded_alpha else None
        record['embedded_alpha_used'] = bool(use_embedded_alpha)
        record['embedded_alpha_sha256'] = hashlib.sha256(alpha.tobytes()).hexdigest() if alpha else None
        record['legacy_alpha_discarded'] = result.image.mode == 'RGBA' and not use_embedded_alpha
        if record['legacy_alpha_discarded']:
            record['warnings'] = list(record['warnings']) + ['Legacy generation decodes opaque RGB; embedded alpha is not used as its segmentation mask. Supply a separate reviewed mask.']
        record['colour_accuracy_verified'] = False
        return (image, record, alpha) if use_embedded_alpha else (image, record)
    finally:
        result.close()
