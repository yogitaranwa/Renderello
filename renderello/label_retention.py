"""Same-coordinate source-pixel and alpha retention checks, not OCR truth.
A correct string cannot be inferred from byte equality or invented by this tool.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .intent_mask import sha


def check_regions(source:Path,cutout:Path,regions:dict):
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds four million pixels')
        rgb=np.array(im.convert('RGB'));size=list(im.size)
    if regions.get('source_sha256')!=sha(source) or regions.get('image_size')!=size:raise ValueError('regions must bind exact source SHA256 and dimensions')
    with Image.open(cutout) as im:
        if im.mode!='RGBA' or list(im.size)!=size:raise ValueError('cutout must be same-coordinate source-sized RGBA; resampled scenes are not supported')
        rgba=np.array(im)
    items=regions.get('regions')
    if not isinstance(items,list) or not 1<=len(items)<=32:raise ValueError('one to32 regions required')
    names=set();results=[]
    for item in items:
        name=item.get('name');b=item.get('box');threshold=item.get('minimum_alpha',255)
        if not isinstance(name,str) or not name.strip() or name in names:raise ValueError('unique nonempty region names required')
        names.add(name)
        if not isinstance(b,list) or len(b)!=4 or any(type(x)!=int for x in b) or not (0<=b[0]<b[2]<=size[0] and 0<=b[1]<b[3]<=size[1]):raise ValueError('region box must be valid integer source coordinates')
        if type(threshold)!=int or not 0<=threshold<=255:raise ValueError('minimum alpha must be integer0-255')
        x1,y1,x2,y2=b;a=rgba[y1:y2,x1:x2,3];diff=np.abs(rgba[y1:y2,x1:x2,:3].astype('int16')-rgb[y1:y2,x1:x2].astype('int16'));changed=np.any(diff,axis=2);lost=a<threshold
        # Evaluate the actual RGB seen after source-over, not only hidden RGBA RGB.
        # Preserve configurable retention semantics; exact compositing is separate.
        rgb_region=rgb[y1:y2,x1:x2]
        region_image=Image.fromarray(rgba[y1:y2,x1:x2], 'RGBA')
        composite_checks={}
        for background,color in [('white',(255,255,255)),('black',(0,0,0)),('magenta',(255,0,255))]:
            bg=Image.new('RGBA',region_image.size,(*color,255))
            seen=np.array(Image.alpha_composite(bg,region_image).convert('RGB'))
            error=np.abs(seen.astype('int16')-rgb_region.astype('int16'))
            composite_checks[background]=dict(changed_pixels=int(np.any(error,axis=2).sum()),max_channel_error=int(error.max()),mean_channel_error=float(error.mean()))
        exact_composite=all(x['changed_pixels']==0 for x in composite_checks.values())
        results.append(dict(name=name,box=b,pixels=int(a.size),source_rgb_exact=not bool(changed.any()),rgb_changed_pixels=int(changed.sum()),rgb_max_error=int(diff.max()),minimum_alpha_observed=int(a.min()),minimum_alpha_required=threshold,below_required_alpha_pixels=int(lost.sum()),retention_check_passed=not bool(changed.any() or lost.any()),composite_checks=composite_checks,source_native_composite_exact=exact_composite))
    return dict(source_sha256=sha(source),cutout_sha256=sha(cutout),regions_sha256=None,image_size=size,regions=results,all_retention_checks_passed=all(x['retention_check_passed'] for x in results),all_source_native_composites_exact=all(x['source_native_composite_exact'] for x in results),quality='NOT_EVALUATED',transcription='NOT_PERFORMED',warnings=['Same-coordinate byte and opacity checks only, not independent reading or verification of label text.','RGBA RGB equality is not visible composite equality. Composite checks use actual Pillow source-over on white, black and magenta.','A source may already be blurry, wrong or incomplete. Region selection and minimum opacity must be reviewed.','Transparent regions should not be declared opaque. This tool does not estimate optical transmission.'])


def check_file(source,cutout,regions_path,out):
    if out.exists():raise ValueError('output exists; preserve previous reports')
    regions=json.loads(regions_path.read_text());report=check_regions(source,cutout,regions);report['regions_sha256']=sha(regions_path);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));return report
