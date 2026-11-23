"""Render an existing RGB photo and supplied alpha on an existing scene.

Offline only. No automatic segmentation, generation, resizing, crop or
physical relighting. Fractional alpha is straight sRGB coverage, not optics.
"""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import os
import stat
import tempfile
import shutil
import numpy as np
from PIL import Image
from .cached_native_scene import compose_cached_native


def snapshot(path, mode, pixels):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('regular native PNG input required')
        data = stream.read(16_000_001)
    if len(data) > 16_000_000:
        raise ValueError('encoded PNG input exceeds16MBbudget')
    # Pillow can truncate16bitRGB PNG silently, so inspect IHDR before decode.
    if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data) < 29 or data[12:16] != b'IHDR' or data[24] != 8:
        raise ValueError('8bit native PNG required, no implicit conversion')
    with Image.open(BytesIO(data)) as image:
        if image.mode != mode or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('single-frame '+mode+' PNG required, no implicit conversion')
        if image.width * image.height > pixels:
            raise ValueError('native input exceeds pixel budget, no resizing')
        if 'transparency' in image.info:
            raise ValueError('native RGB/L PNG must not carry hidden color-key transparency')
        if image.getexif().get(274, 1) != 1 or image.info.get('icc_profile'):
            raise ValueError('normalize orientation/profile before pairing reviewed pixels')
        image.load()
        array = np.array(image)
    return data, array


def run(source, alpha, scene, out, x, y):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('preserve earlier output; choose a new directory')
    inputs = {}
    for name, path, mode, budget in [('source',source,'RGB',1_000_000),
                                     ('alpha',alpha,'L',1_000_000),
                                     ('scene',scene,'RGB',2_000_000)]:
        inputs[name] = snapshot(path, mode, budget)
    result, report = compose_cached_native(inputs['source'][1], inputs['alpha'][1],
                                           inputs['scene'][1], translation=(x,y))
    # Calculate and validate everything before creating a staged result.
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+out.name+'-staging-', dir=out.parent))
    try:
        for name, (data, _) in inputs.items():
            (stage/(name+'.png')).write_bytes(data)
        Image.fromarray(result).save(stage/'composite.png')
        manifest = dict(schema='offline-native-scene-result.v1',
            status='EXECUTED_REVIEW_REQUIRED', photo_accepted=False,
            automatic_segmentation=False, provider_calls=0, credits_spent=0,
            source_bytes_preserved=True, scene_bytes_preserved=True,
            input_file_sha256={name:hashlib.sha256(data).hexdigest() for name,(data,_) in inputs.items()},
            output_sha256=hashlib.sha256((stage/'composite.png').read_bytes()).hexdigest(),
            report=report)
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
        # Reserve the destination rather than replacing a newly appeared result.
        # Copy into the reserved directory with an incomplete marker. On failure
        # leave evidence, not a misleading success manifest.
        out.mkdir()
        (out/'INCOMPLETE').touch()
        for path in stage.iterdir():
            if path.name != 'manifest.json': shutil.copyfile(path,out/path.name)
        shutil.copyfile(stage/'manifest.json',out/'manifest.json')
        (out/'INCOMPLETE').unlink()
        return manifest
    finally:
        shutil.rmtree(stage)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','alpha','scene','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--x',type=int,required=True)
    parser.add_argument('--y',type=int,required=True)
    args=parser.parse_args(argv)
    result=run(**vars(args))
    print(json.dumps(result,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
