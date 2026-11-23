"""Explicit source/mask coordinate preparation. Not enabled by legacy routes.

A prepared image is a derivative, not an invented/reconstructed product. Existing
alpha and caller masks are not evidence of semantic segmentation correctness.
"""
from pathlib import Path
import hashlib,json,tempfile,shutil
from PIL import Image
from .image_intake import inspect_image,IntakePolicy,IntakeError,_read_bounded

_TRANSPOSE={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,
 4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,
 6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}

def prepare_reviewed_source(source,destination,*,mask=None,mask_coordinates=None,policy=None):
 """Create a new normalized derivative and reviewed-coordinate mask.

 For an external mask, coordinates must explicitly be 'encoded' (raw pixel
 grid before EXIF) or 'display' (normalized display orientation). Width/height
 alone cannot disambiguate flips,180-degree turns,or square images. An external
 mask conflicting with source alpha is refused,never multiplied/replaced.
 """
 dest=Path(destination)
 if dest.exists():raise FileExistsError('Preserve previous prepared sources')
 if mask is None and mask_coordinates is not None:raise ValueError('Mask coordinates without mask')
 if mask is not None and mask_coordinates not in ('encoded','display'):raise ValueError('Declare encoded or display mask coordinates')
 p=policy or IntakePolicy();r=inspect_image(source,p)
 try:
  image=r.image
  alpha=image.getchannel('A') if image.mode=='RGBA' else None
  mask_hash=None
  if mask is not None:
   data=_read_bounded(Path(mask),p.max_bytes);mask_hash=hashlib.sha256(data).hexdigest()
   # Pillow can expose a 16-bit grayscale PNG as I; reject before decode.
   if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)<29 or data[12:16]!=b'IHDR' or data[24]!=8:raise IntakeError('Mask must be an 8-bit PNG, no silent depth conversion')
   from io import BytesIO
   with Image.open(BytesIO(data)) as m:
    # External mask carries no hidden orientation or colour interpretation.
    if m.format!='PNG' or m.mode!='L' or getattr(m,'n_frames',1)!=1:raise IntakeError('Mask must be single-frame 8-bit grayscale PNG')
    if m.getexif().get(274,1)!=1 or m.info.get('icc_profile') or 'transparency' in m.info:raise IntakeError('Mask metadata is ambiguous; use plain grayscale coordinates')
    expected=tuple(r.record['input_size'] if mask_coordinates=='encoded' else r.record['output_size'])
    if m.size!=expected:raise IntakeError('Mask dimensions disagree with declared coordinates')
    m.load();matte=m.copy()
   o=r.record['orientation']
   if mask_coordinates=='encoded' and o in _TRANSPOSE:matte=matte.transpose(_TRANSPOSE[o])
   if alpha is not None and matte.tobytes()!=alpha.tobytes():raise IntakeError('External mask conflicts with embedded source alpha')
   alpha=matte
  if alpha is None:raise IntakeError('No reviewed alpha: prepare an explicit mask or provide embedded alpha')
  if alpha.getextrema()[1]==0:raise IntakeError('Empty alpha cannot represent a visible product')
  # Staging prevents a complete-looking output after decode/write failure.
  dest.parent.mkdir(parents=True,exist_ok=True)
  stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
  try:
   rgb=image.convert('RGB');rgb.info.clear();alpha.info.clear()
   rgb.save(stage/'normalized-source.png');alpha.save(stage/'reviewed-alpha.png')
   for path,ref in [(stage/'normalized-source.png',rgb),(stage/'reviewed-alpha.png',alpha)]:
    with Image.open(path) as check:
     check.load()
     if check.size!=ref.size or check.mode!=ref.mode or check.tobytes()!=ref.tobytes():raise IntakeError('Prepared PNG round-trip mismatch')
   record=dict(intake=r.record,mask_coordinates=mask_coordinates,external_mask_sha256=mask_hash,
    source_png_sha256=hashlib.sha256((stage/'normalized-source.png').read_bytes()).hexdigest(),
    alpha_png_sha256=hashlib.sha256((stage/'reviewed-alpha.png').read_bytes()).hexdigest(),
    semantic_mask_accuracy='NOT_VERIFIED',publication_status='NOT_EVALUATED',legacy_routes_changed=False)
   (stage/'prepared-source.json').write_text(json.dumps(record,indent=2))
   dest.mkdir()
   try:
    for child in stage.iterdir():child.rename(dest/child.name)
   except BaseException:
    (dest/'PREPARATION_INCOMPLETE').touch();raise
   return record
  finally:
   # Only the agent-created staging directory, never a caller's folder.
   if stage.exists():shutil.rmtree(stage)
 finally:r.close()


def main(argv=None):
 import argparse
 p=argparse.ArgumentParser(description='Prepare a separate normalized source and explicitly aligned mask; legacy generation defaults stay unchanged')
 p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 p.add_argument('--mask',type=Path);p.add_argument('--mask-coordinates',choices=['encoded','display'])
 p.add_argument('--allow-colour-conversion',action='store_true')
 a=p.parse_args(argv)
 result=prepare_reviewed_source(a.source,a.out,mask=a.mask,mask_coordinates=a.mask_coordinates,policy=IntakePolicy(allow_colour_conversion=a.allow_colour_conversion))
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
