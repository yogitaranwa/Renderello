"""Opt-in bridge: verified estimated alpha with original normalized RGB only.

This does not accept estimated foreground colors or certify mask semantics.
Legacy composition and defaults are unchanged.
"""
from pathlib import Path
from io import BytesIO
import hashlib,json,tempfile,shutil
import numpy as np
from PIL import Image
from .image_intake import IntakeError,IntakePolicy,_read_bounded,inspect_image
from .reviewed_source import prepare_reviewed_source


def prepare_alpha_only(source,bundle,destination):
    source,bundle,dest=Path(source),Path(bundle),Path(destination)
    if dest.exists() or dest.is_symlink():raise FileExistsError('Preserve previous prepared sources')
    if bundle.is_symlink() or not bundle.is_dir():raise IntakeError('Regular matting bundle directory required')
    def read(name):
        p=bundle/name
        if p.is_symlink() or not p.is_file():raise IntakeError('Regular bundle file required: '+name)
        return _read_bounded(p,IntakePolicy().max_bytes)
    manifest_raw=read('matting-manifest.json')
    try:m=json.loads(manifest_raw)
    except (ValueError,TypeError) as e:raise IntakeError('Invalid matting manifest') from e
    if not isinstance(m,dict) or not isinstance(m.get('output_sha256'),dict):raise IntakeError('Matting output hashes required')
    raw=_read_bounded(source,IntakePolicy().max_bytes)
    if hashlib.sha256(raw).hexdigest()!=m.get('source_sha256'):raise IntakeError('Original source hash mismatch')
    files={}
    for name in ['source.png','refined-alpha.png','trimap.png']:
        data=read(name)
        if hashlib.sha256(data).hexdigest()!=m['output_sha256'].get(name):raise IntakeError('Matting bundle hash mismatch: '+name)
        files[name]=data
    # Decode the exact bytes whose hash was checked, never reopen a mutable
    # external source path after validation. Close before opening on Windows.
    with tempfile.TemporaryDirectory(prefix='renderello-matting-source-') as td:
        snapshot=Path(td)/'source.png'
        snapshot.write_bytes(raw)
        original=inspect_image(snapshot)
    try:
        # The matting implementation works on encoded pixels. Do not guess an
        # EXIF transform or silently reinterpret a raw/normalized coordinate map.
        if original.record['orientation']!=1 or original.record['input_mode']!='RGB' or original.record['transformations']:
            raise IntakeError('Use normalized RGB source with no orientation or color transform')
        if original.image.width*original.image.height>1_000_000:raise IntakeError('Matting bridge capped at one million source pixels; no resizing')
        rgb=np.array(original.image);size=original.image.size
    finally:original.close()
    arrays={}
    for name,mode in [('source.png','RGB'),('refined-alpha.png','L'),('trimap.png','L')]:
        data=files[name]
        if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)<29 or data[24]!=8:raise IntakeError('Plain 8-bit PNG required')
        with Image.open(BytesIO(data)) as im:
            if im.format!='PNG' or im.mode!=mode or im.size!=size or getattr(im,'n_frames',1)!=1:raise IntakeError('Matting bundle size/mode mismatch')
            if im.getexif().get(274,1)!=1 or im.info.get('icc_profile') or 'transparency' in im.info:raise IntakeError('Ambiguous matting metadata')
            im.load();arrays[name]=np.array(im)
    if not np.array_equal(arrays['source.png'],rgb):raise IntakeError('Matting pixels disagree with original source')
    tri,alpha=arrays['trimap.png'],arrays['refined-alpha.png']
    if not set(np.unique(tri)).issubset({0,128,255}) or not (np.any(tri==0) and np.any(tri==255)):raise IntakeError('Valid known foreground/background trimap required')
    if np.any(alpha[tri==255]!=255) or np.any(alpha[tri==0]!=0):raise IntakeError('Refined alpha conflicts with protected trimap core')
    evidence=dict(method='verified_matting_alpha_only',source_sha256=hashlib.sha256(raw).hexdigest(),matting_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),refined_alpha_sha256=hashlib.sha256(files['refined-alpha.png']).hexdigest(),source_rgb_policy='ORIGINAL_RGB_ONLY_ESTIMATED_FOREGROUND_IGNORED',protected_core_pixels=int((tri==255).sum()),alpha_semantics='ESTIMATED_NOT_GROUND_TRUTH',physical_transmission='NOT_INFERRED',quality_acceptance='NOT_EVALUATED',defaults_changed=False)
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-alpha-',dir=dest.parent))
    try:
        # Copy verified mask bytes, not an unchecked path read again later.
        (stage/'alpha.png').write_bytes(files['refined-alpha.png'])
        (stage/'source.png').write_bytes(raw)
        record=prepare_reviewed_source(stage/'source.png',stage/'prepared',mask=stage/'alpha.png',mask_coordinates='display')
        record['estimated_alpha_provenance']=evidence
        (stage/'prepared/prepared-source.json').write_text(json.dumps(record,indent=2))
        # Never overwrite a destination that appeared while validating.
        dest.mkdir()
        try:
            for child in (stage/'prepared').iterdir():child.rename(dest/child.name)
        except BaseException:
            (dest/'PREPARATION_INCOMPLETE').touch();raise
        return record
    finally:shutil.rmtree(stage)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(argv);print(json.dumps(prepare_alpha_only(a.source,a.bundle,a.out),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
