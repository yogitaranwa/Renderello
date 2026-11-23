"""Free native-pixel graphic variants, explicitly not photographic scenes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

PRESETS = ('navy_diagonal', 'cream_editorial', 'cobalt_circle', 'sage_frame')

def render_variants(source, rgba_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior variant runs; use a new directory')
    with Image.open(source) as im:
        if im.mode != 'RGB' or im.getexif().get(274, 1) != 1 or getattr(im,'n_frames',1) != 1:
            raise ValueError('Single orientation-normalized RGB source required')
        rgb = np.array(im)
        profile = im.info.get('icc_profile')
    with Image.open(rgba_path) as im:
        if im.mode != 'RGBA' or im.size != (rgb.shape[1], rgb.shape[0]):
            raise ValueError('Exact native RGBA dimensions required')
        rgba = np.array(im)
    if not np.array_equal(rgb, rgba[:, :, :3]):
        raise ValueError('RGBA repaints source pixels')
    a = rgba[:, :, 3]
    yy, xx = np.where(a > 0)
    if len(xx) == 0:
        raise ValueError('Empty predicted mask')
    if rgb.shape[0] * rgb.shape[1] > 1_000_000:
        raise ValueError('One-megapixel source budget')
    box = (xx.min(), yy.min(), xx.max()+1, yy.max()+1)
    crop = rgba[box[1]:box[3], box[0]:box[2]]
    ph, pw = crop.shape[:2]
    w, h = max(800, pw+48), max(800, ph+180)
    if w*h > 2_000_000:
        raise ValueError('Canvas exceeds bounded native variant budget')
    output.mkdir(parents=True)
    rows = []
    for key in PRESETS:
        bg = Image.new('RGB', (w,h), '#f7f3ed')
        d = ImageDraw.Draw(bg)
        if key == 'navy_diagonal':
            d.rectangle((0,0,w,h), fill='#142236')
            d.polygon([(0,int(h*.25)),(w,int(h*.06)),(w,int(h*.86)),(0,h)], fill='#d9e0e7')
            d.line([(int(w*.06),int(h*.11)),(int(w*.41),int(h*.11))], fill='#cb9b58', width=5)
        elif key == 'cream_editorial':
            d.rectangle((int(w*.08),int(h*.08),int(w*.92),int(h*.92)), fill='#e5d8c6')
            d.rectangle((int(w*.13),int(h*.13),int(w*.87),int(h*.87)), outline='#9e805d',width=2)
            d.rectangle((0,int(h*.73),w,h), fill='#f6eee3')
        elif key == 'cobalt_circle':
            d.rectangle((0,0,w,h), fill='#183a87')
            d.ellipse((int(-w*.10),int(h*.12),int(w*1.10),int(h*1.24)), fill='#b8d6ed')
            d.rectangle((int(w*.77),0,int(w*.84),int(h*.21)), fill='#e4a48b')
        else:
            d.rectangle((0,0,w,h), fill='#c8d4c5')
            d.rectangle((int(w*.07),int(h*.07),int(w*.93),int(h*.93)), fill='#edf0e8')
            d.rectangle((0,int(h*.79),w,h), fill='#6f8978')
            d.ellipse((int(w*.72),int(h*.07),int(w*.91),int(h*.26)), fill='#d6b98c')
        pixels = np.array(bg)
        px, py = (w-pw)//2, (h-ph)//2+15
        region = pixels[py:py+ph, px:px+pw]
        alpha = crop[:,:,3:4].astype(np.float64)/255
        region[:] = np.rint(crop[:,:,:3]*alpha+region*(1-alpha)).clip(0,255).astype('uint8')
        opaque = crop[:,:,3] == 255
        changed = int(np.count_nonzero(np.any(region[opaque] != crop[:,:,:3][opaque],axis=1)))
        assert changed == 0
        meta = PngImagePlugin.PngInfo()
        meta.add_text('status', 'DESIGNED_CPU_GRAPHIC_NOT_PHOTOGRAPHIC_NOT_ACCEPTED')
        path = output/(key+'.png')
        Image.fromarray(pixels).save(path,pnginfo=meta,**({'icc_profile':profile} if profile else {}))
        rows.append(dict(preset=key,output=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),opaque_changed_pixels=changed,placement=[px,py,pw,ph]))
    record = dict(schema='renderello-native-graphic-variants-v1',source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),rgba_sha256=hashlib.sha256(Path(rgba_path).read_bytes()).hexdigest(),accepted=0,source_icc_preserved=bool(profile),source_icc_sha256=hashlib.sha256(profile).hexdigest() if profile else None,physical_relighting=False,physical_contact=False,product_resampled=False,quality='NOT_EVALUATED',mask_provenance='IMPORTED_MODEL_ESTIMATE_NOT_GROUND_TRUTH',variants=rows)
    (output/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--rgba',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(render_variants(a.source,a.rgba,a.out),indent=2))
