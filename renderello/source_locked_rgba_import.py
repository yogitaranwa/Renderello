"""Import a pinned external RGBA without accepting its foreground or scene quality.

The file hash binds supplied evidence; it does not authenticate the model or its
publisher. Exact source RGB is required even under zero alpha. No resize,
repainting, mask enforcement, or automatic candidate selection happens here.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import re
import os
import stat
import zlib
import numpy as np
from PIL import Image
from .candidate_diagnostics import native_rgb, digest


def _strict_png(data):
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Plain 8-bit RGBA PNG required')
    pos=8;count=0;idat=False
    interpretation={b'iCCP',b'gAMA',b'sRGB',b'cHRM',b'cICP',b'mDCV',b'cLLI',b'eXIf',b'tRNS',b'acTL',b'fcTL',b'fdAT'}
    while pos<len(data):
        count+=1
        if count>4096 or len(data)-pos<12:raise ValueError('Bounded complete PNG chunks required')
        length=int.from_bytes(data[pos:pos+4],'big');kind=data[pos+4:pos+8];end=pos+12+length
        if end>len(data):raise ValueError('Truncated PNG chunk')
        payload=data[pos+8:pos+8+length];crc=int.from_bytes(data[pos+8+length:end],'big')
        if zlib.crc32(kind+payload)&0xffffffff!=crc:raise ValueError('PNG chunk checksum mismatch')
        if count==1:
            if kind!=b'IHDR' or length!=13 or payload[8]!=8 or payload[9]!=6:
                raise ValueError('Explicit 8-bit RGBA PNG header required; no depth conversion')
        elif kind==b'IHDR':raise ValueError('Duplicate PNG header')
        if kind in interpretation:raise ValueError('PNG colour/orientation/transparency/frame interpretation metadata refused')
        if kind==b'IDAT':idat=True
        if kind==b'IEND':
            if length or end!=len(data)or not idat:raise ValueError('Exact complete PNG stream required')
            return
        pos=end
    raise ValueError('PNG end marker required')


def load_source_locked_rgba(path, source_rgb, expected_sha256):
    native_rgb(source_rgb)
    if source_rgb.shape[0] * source_rgb.shape[1] > 1_000_000:
        raise ValueError('External RGBA exceeds one-megapixel replay budget')
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('Lowercase SHA-256 of the encoded result is required')
    path = Path(path)
    if path.is_symlink():raise ValueError('Regular bounded RGBA result required')
    flags=os.O_RDONLY|getattr(os,'O_NONBLOCK',0)|getattr(os,'O_NOFOLLOW',0)
    with os.fdopen(os.open(path,flags),'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode)or info.st_size>20_000_000:
            raise ValueError('Regular bounded RGBA result required')
        data=stream.read(20_000_001)
    if len(data) > 20_000_000 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('External RGBA size or SHA-256 mismatch')
    _strict_png(data)
    with Image.open(BytesIO(data)) as im:
        if im.format != 'PNG' or im.mode != 'RGBA' or im.size != (source_rgb.shape[1], source_rgb.shape[0]):
            raise ValueError('Native-size RGBA PNG required; no conversion or resize')
        if getattr(im, 'n_frames', 1) != 1 or im.info.get('icc_profile') or im.getexif().get(274, 1) != 1:
            raise ValueError('Single normalized unprofiled RGBA frame required')
        rgba = np.array(im)
    changed = int(np.count_nonzero(np.any(rgba[:, :, :3] != source_rgb, axis=2)))
    if changed:
        raise ValueError(f'External RGBA repaints {changed} source RGB pixels')
    alpha = rgba[:, :, 3].copy()
    return alpha, dict(schema='source-locked-rgba-import.v1', encoded_sha256=expected_sha256,
        source_pixels_sha256=digest(source_rgb), imported_alpha_sha256=digest(alpha),
        source_rgb_exact=True, source_rgb_changed_pixels=0, no_new_inference=True,
        model_provenance='NOT_AUTHENTICATED_BY_PIXEL_REPLAY', automatic_acceptance=False,
        physical_relighting=False, glass_optics_verified=False)
