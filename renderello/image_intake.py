"""Opt-in image intake, not automatic product/quality acceptance.

Original bytes are never rewritten. Colour-changing conversions require explicit
policy; unknown profiles and high-bit-depth images fail rather than being silently
truncated. Image identity, material and print readability still need visual review.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import stat
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


class IntakeError(ValueError):
    """An input cannot be safely interpreted under the selected policy."""


@dataclass(frozen=True)
class IntakePolicy:
    max_bytes: int = 40_000_000
    max_pixels: int = 40_000_000
    max_side: int = 16000
    max_aspect: float = 40.0
    allow_colour_conversion: bool = False
    formats: tuple[str, ...] = ('JPEG', 'PNG', 'WEBP', 'TIFF', 'BMP')

    def __post_init__(self):
        for name in ('max_bytes', 'max_pixels', 'max_side'):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                raise ValueError(name + ' must be a positive integer')
        if not math.isfinite(self.max_aspect) or self.max_aspect < 1:
            raise ValueError('max_aspect must be finite and at least one')
        if not self.formats or any(not isinstance(x, str) for x in self.formats):
            raise ValueError('formats must contain image format names')


@dataclass
class IntakeResult:
    image: Image.Image
    record: dict

    def close(self):
        self.image.close()


def _read_bounded(path: Path, limit: int) -> bytes:
    # A stream read enforces the limit even if a file grows after stat().
    if not stat.S_ISREG(path.stat().st_mode):
        raise IntakeError('Input must be a regular file')
    # Nonblocking open plus descriptor recheck closes the FIFO replacement race.
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise IntakeError('Input must be a regular file')
        if os.fstat(f.fileno()).st_size > limit:
            raise IntakeError('Encoded input exceeds byte limit')
        data = f.read(limit + 1)
    if len(data) > limit:
        raise IntakeError('Encoded input exceeds byte limit')
    return data


def _dimensions(image: Image.Image, p: IntakePolicy):
    w, h = image.size
    if min(w, h) < 1 or max(w, h) > p.max_side or w * h > p.max_pixels:
        raise IntakeError('Image dimensions exceed configured budget')
    if max(w / h, h / w) > p.max_aspect:
        raise IntakeError('Extreme aspect ratio requires a separate review path')


def inspect_image(path: str | Path, policy: IntakePolicy | None = None) -> IntakeResult:
    """Decode and normalize orientation only; never invent missing product pixels.

    Returned alpha is retained including transparent-pixel RGB. Converting colour
    space can change RGB, and is explicitly reported rather than called exact.
    Animated and multipage images are refused rather than picking a hidden frame.
    """
    p = policy or IntakePolicy()
    data = _read_bounded(Path(path), p.max_bytes)
    if not data:
        raise IntakeError('Empty input')
    # Pillow exposes 16-bit RGB PNG as RGB after truncation. Inspect IHDR first.
    if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 29:
        if data[12:16] != b'IHDR' or data[24] > 8:
            raise IntakeError('High-bit-depth PNG needs an explicit tone-mapping path')
    digest = hashlib.sha256(data).hexdigest()
    transformations = []
    warnings = []
    try:
        with Image.open(BytesIO(data)) as verifier:
            _dimensions(verifier, p)
            verifier.verify()
        with Image.open(BytesIO(data)) as source:
            fmt, mode, original_size = source.format, source.mode, source.size
            if fmt not in p.formats:
                raise IntakeError('Unsupported format: ' + str(fmt))
            _dimensions(source, p)
            if fmt == 'TIFF':
                bits = source.tag_v2.get(258, (8,))
                bits = (bits,) if isinstance(bits, int) else bits
                if any(b > 8 for b in bits):
                    raise IntakeError('High-bit-depth TIFF needs an explicit tone-mapping path')
            if getattr(source, 'n_frames', 1) != 1:
                raise IntakeError('Animated or multipage input requires explicit frame selection')
            # Do not quietly clamp 16-bit/float/HDR values into 8-bit RGB.
            if mode not in ('RGB', 'RGBA', 'L', 'LA', 'P', 'CMYK'):
                raise IntakeError('Unsupported bit depth or colour mode: ' + mode)
            orientation = source.getexif().get(274, 1)
            if orientation not in range(1, 9):
                raise IntakeError('Invalid EXIF orientation')
            profile = source.info.get('icc_profile')
            if profile and len(profile) > 4_000_000:
                raise IntakeError('ICC profile exceeds size limit')
            # Structural verification used a fresh stream before metadata access.
        with Image.open(BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            if orientation != 1:
                transformations.append('exif_orientation_' + str(orientation))
            _dimensions(image, p)
            alpha = None
            # PNG can carry tRNS colour-key transparency in RGB and L modes,
            # not only palette images. Extract it before RGB/ICC normalization.
            if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                alpha = image.convert('RGBA').getchannel('A')
            if profile:
                if not p.allow_colour_conversion:
                    image.close()
                    raise IntakeError('Embedded ICC profile requires explicit colour-conversion policy')
                try:
                    src_profile = ImageCms.ImageCmsProfile(BytesIO(profile))
                    dst_profile = ImageCms.createProfile('sRGB')
                    # alpha is passed separately; LittleCMS consumes colour channels.
                    base = image.convert('RGB') if image.mode in ('RGBA', 'P') else image
                    image2 = ImageCms.profileToProfile(base, src_profile, dst_profile, outputMode='RGB')
                except (ValueError, OSError, ImageCms.PyCMSError) as e:
                    image.close()
                    raise IntakeError('ICC conversion failed; no unprofiled fallback') from e
                image.close()
                image = image2
                transformations.append('icc_to_srgb')
            elif mode == 'CMYK':
                image.close()
                raise IntakeError('Unprofiled CMYK is ambiguous; supply a colour-managed source')
            else:
                image2 = image.convert('RGB')
                image.close()
                image = image2
                if mode in ('L', 'LA', 'P'):
                    transformations.append('lossless_8bit_channel_expansion')
                warnings.append('No ICC profile: RGB treated as sRGB, not colour-accuracy verified')
            if alpha is not None:
                image.putalpha(alpha)
                alpha.close()
            # Do not carry source EXIF or an obsolete ICC tag into normalized output.
            image.info.clear()
            if min(image.size) < 512:
                warnings.append('Low-resolution source: no upscaling or detail recovery performed')
            if max(image.size) / min(image.size) > 8:
                warnings.append('Wide/tall source: framing and template compatibility need review')
            if image.mode == 'RGBA':
                warnings.append('Existing alpha retained; semantic mask correctness is not verified')
            record = dict(schema_version=1, source_sha256=digest,
                          encoded_bytes=len(data), input_format=fmt, input_mode=mode,
                          input_size=list(original_size), output_size=list(image.size),
                          output_mode=image.mode, orientation=orientation,
                          icc_present=bool(profile), transformations=transformations,
                          warnings=warnings, automatic_publication_pass=False,
                          product_identity='NOT_EVALUATED', print_accuracy='NOT_EVALUATED',
                          glass_optics='NOT_EVALUATED', policy=asdict(p))
            return IntakeResult(image, record)
    except IntakeError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as e:
        raise IntakeError('Image decoding/metadata validation failed: ' + type(e).__name__) from e


def save_intake(result: IntakeResult, destination: str | Path) -> Path:
    """Stage verified output in a new directory. Never replace previous outputs.

    A temporary sibling prevents a half-written PNG from looking like completion.
    Caller owns destination choice; metadata is private unless explicitly shared.
    """
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError('Choose a new output directory')
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + dest.name + '-', dir=dest.parent))
    try:
        result.image.save(stage / 'normalized.png', format='PNG')
        with Image.open(stage / 'normalized.png') as check:
            check.load()
            if check.mode != result.image.mode or check.size != result.image.size or check.tobytes() != result.image.tobytes():
                raise IntakeError('PNG round-trip differs from normalized pixels')
        record = dict(result.record)
        record['normalized_png_sha256'] = hashlib.sha256((stage / 'normalized.png').read_bytes()).hexdigest()
        (stage / 'intake.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        # mkdir protects existing destinations, unlike an overwriting rename.
        dest.mkdir()
        try:
            for child in stage.iterdir():
                child.rename(dest / child.name)
        except BaseException:
            # A failure marker distinguishes partial output from completed intake.
            (dest / 'INTAKE_INCOMPLETE').touch()
            raise
        return dest
    finally:
        # Only our own empty temporary directory is removed, never user files.
        if stage.exists() and not any(stage.iterdir()):
            stage.rmdir()


def main(argv=None):
    """Explicit local CLI; this does not change existing generation defaults."""
    import argparse
    parser = argparse.ArgumentParser(description='Inspect and normalize a source derivative without overwriting the source')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-colour-conversion', action='store_true')
    args = parser.parse_args(argv)
    result = inspect_image(args.source, IntakePolicy(allow_colour_conversion=args.allow_colour_conversion))
    try:
        save_intake(result, args.out)
        print(json.dumps(result.record, indent=2))
    finally:
        result.close()


if __name__ == '__main__':
    main()
