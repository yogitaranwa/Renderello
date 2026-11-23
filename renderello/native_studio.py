"""Designed 2D studio candidates with native-size source-exact opaque product cores.
Procedural illustration, not photographed scenery, recovered lighting or pose.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from .intent_mask import sha

STYLES={
 'porcelain_sweep':dict(name='Porcelain sweep',top=(236,234,227),bottom=(247,247,244),accent=(217,214,203),shape='sweep'),
 'sage_arch':dict(name='Sage arch',top=(143,161,148),bottom=(201,211,194),accent=(225,228,213),shape='arch'),
 'clay_plinth':dict(name='Clay plinth',top=(199,151,127),bottom=(226,188,159),accent=(167,115,94),shape='plinth'),
 'noir_halo':dict(name='Noir halo',top=(23,28,32),bottom=(54,61,65),accent=(110,120,123),shape='halo'),
 'amber_paper':dict(name='Amber paper',top=(223,176,94),bottom=(239,204,136),accent=(183,127,56),shape='paper'),
 'linen_daylight':dict(name='Linen daylight',top=(218,208,189),bottom=(238,232,216),accent=(167,154,129),shape='window'),
 'cobalt_aperture':dict(name='Cobalt aperture',top=(31,58,106),bottom=(61,96,145),accent=(151,182,203),shape='aperture'),
 'blush_steps':dict(name='Blush steps',top=(219,181,177),bottom=(239,211,204),accent=(185,138,133),shape='steps'),
}


def background(style,size=(1024,1024),seed=0,baseline=860):
    if style not in STYLES:raise ValueError('unknown studio style')
    w,h=size
    if type(w)!=int or type(h)!=int or not (512<=w<=2048 and 512<=h<=2048):raise ValueError('canvas dimensions must be512-2048')
    spec=STYLES[style];y,x=np.mgrid[0:h,0:w];t=y/max(1,h-1);top=np.array(spec['top']);bottom=np.array(spec['bottom']);a=top[None,None,:]*(1-t[:,:,None])+bottom[None,None,:]*t[:,:,None]
    # Gentle broad illumination and fixed-seed subpixel paper grain only in scenery.
    light=np.exp(-(((x/w-.35)/.6)**2+((y/h-.35)/.7)**2));a+=light[:,:,None]*7
    a+=np.random.default_rng(seed).normal(0,.55,(h,w,1));im=Image.fromarray(np.clip(a,0,255).astype('uint8'));d=ImageDraw.Draw(im);c=spec['accent'];shape=spec['shape'];sx=w/1024;sy=h/1024
    def box(v):return tuple(round(n*(sx if i%2==0 else sy)) for i,n in enumerate(v))
    if shape=='sweep':
        # No hard horizon; a wide soft neutral floor roll.
        layer=Image.new('RGBA',(w,h));ld=ImageDraw.Draw(layer);ld.ellipse(box((-400,750,1500,1450)),fill=(*c,90));layer=layer.filter(ImageFilter.GaussianBlur(80*sx));im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')
    elif shape=='arch':
        d.rounded_rectangle(box((145,110,880,1250)),radius=round(360*sx),fill=c);d=ImageDraw.Draw(im);d.rectangle(box((145,500,880,1250)),fill=c)
    elif shape=='plinth':
        d.polygon(box((230,860,760,860,960,1150,80,1150)),fill=c);d.polygon(box((230,860,760,860,790,885,205,885)),fill=(231,190,162))
    elif shape=='halo':
        arr=np.array(im).astype(float);v=np.exp(-(((x/w-.5)/.30)**2+((y/h-.47)/.34)**2));arr+=v[:,:,None]*40;im=Image.fromarray(np.clip(arr,0,255).astype('uint8'))
    elif shape=='paper':
        d.polygon(box((0,260,1024,40,1024,700,0,920)),fill=(236,194,117));layer=Image.new('RGBA',(w,h));ld=ImageDraw.Draw(layer);ld.polygon(box((0,260,1024,40,1024,85,0,305)),fill=(*c,50));im=Image.alpha_composite(im.convert('RGBA'),layer.filter(ImageFilter.GaussianBlur(12*sx))).convert('RGB')
    elif shape=='window':
        layer=Image.new('RGBA',(w,h));ld=ImageDraw.Draw(layer)
        for shift in [0,270,540]:ld.polygon(box((-250+shift,0,-160+shift,0,500+shift,1024,410+shift,1024)),fill=(*c,45))
        layer=layer.filter(ImageFilter.GaussianBlur(16*sx));im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')
    elif shape=='aperture':
        d.rounded_rectangle(box((180,120,845,970)),radius=round(300*sx),fill=c);d.rounded_rectangle(box((215,155,810,970)),radius=round(265*sx),fill=(90,129,167))
    elif shape=='steps':
        d.rectangle(box((80,900,944,1024)),fill=c);d.rectangle(box((180,baseline,844,900)),fill=(226,188,182));d.rectangle(box((180,baseline,844,baseline+10)),fill=(242,216,206))
    return im


def load_product(bundle):
    m=json.loads((bundle/'matting-manifest.json').read_text())
    if m.get('source_rgb_unchanged') is not True or m.get('edge_rgb_estimated') is not False:raise ValueError('native studio requires RGB-unchanged matting bundle, not estimated edge colors')
    for f in ('source.png','refined-alpha.png','refined-cutout.png'):
        if sha(bundle/f)!=m.get('output_sha256',{}).get(f):raise ValueError('matting bundle output hash mismatch: '+f)
    with Image.open(bundle/'source.png') as im:rgb=np.array(im.convert('RGB'))
    with Image.open(bundle/'refined-cutout.png') as im:
        if im.mode!='RGBA':raise ValueError('RGBA cutout required')
        rgba=np.array(im)
    with Image.open(bundle/'refined-alpha.png') as im:
        if im.mode!='L':raise ValueError('L alpha required')
        alpha=np.array(im)
    if rgba.shape[:2]!=rgb.shape[:2] or not np.array_equal(rgba[:,:,:3],rgb) or not np.array_equal(rgba[:,:,3],alpha):raise ValueError('cutout RGB/alpha inconsistent')
    if not np.any(alpha==255):raise ValueError('opaque product core required')
    yy,xx=np.where(alpha>0)
    if not len(xx):raise ValueError('empty alpha')
    # The one-pixel border must be background. This is a mechanical warning
    # boundary, not a proof the source framing is complete.
    if np.any(alpha[0]) or np.any(alpha[-1]) or np.any(alpha[:,0]) or np.any(alpha[:,-1]):raise ValueError('mask touches source frame; complete source needs review before native studio')
    return rgba,(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)),m


def compose_native(bundle:Path,style,out:Path,*,source_pose,material,framing,regions_path=None):
    if out.exists():raise ValueError('output exists; preserve earlier candidates')
    if source_pose not in ('front','elevated'):raise ValueError('front or elevated studio source declaration required; no pose change')
    if material!='opaque' or framing!='complete':raise ValueError('opaque complete source required; no transparent paste or uncropping')
    rgba,bbox,source_meta=load_product(bundle);x1,y1,x2,y2=bbox;crop=rgba[y1:y2,x1:x2];ph,pw=crop.shape[:2]
    if ph>760 or pw>760:raise ValueError('native product exceeds760px; no silent resizing. Use a smaller source or larger authored workflow explicitly')
    if (style=='clay_plinth' and pw>450) or (style=='blush_steps' and pw>580):raise ValueError('native product too wide for this designed platform; choose a sweep or halo without a narrow support')
    left=(1024-pw)//2
    baseline=875 if style=='clay_plinth' else (865 if style=='blush_steps' else 860)
    support_bottom=int(np.where(crop[:,:,3]>127)[0].max())+1
    top=baseline-support_bottom
    bg=background(style);alpha=Image.fromarray(crop[:,:,3]);sh=Image.new('L',(1024,1024));sd=ImageDraw.Draw(sh)
    sd.ellipse((left-10,baseline-5,left+pw+25,baseline+9),fill=70)
    sh=sh.filter(ImageFilter.GaussianBlur(10));scene=Image.composite(Image.new('RGB',bg.size,(16,17,18)),bg,sh)
    # One source-native translation, never scale, rotate, warp or relight product.
    scene.paste(Image.fromarray(crop[:,:,:3]),(left,top),alpha);result=np.array(scene);placed=result[top:top+ph,left:left+pw];core=crop[:,:,3]==255
    error=int(np.abs(placed[core].astype('int16')-crop[:,:,:3][core].astype('int16')).max())
    if error:raise ValueError('native opaque core changed unexpectedly')
    out.mkdir(parents=True);bg.save(out/'background.png');sh.save(out/'shadow-alpha.png');scene.save(out/'candidate.png');Image.fromarray(crop).save(out/'native-product-crop.png')
    report=dict(style=style,name=STYLES[style]['name'],scene_kind='designed_2D_procedural_studio_candidate_NOT_photographed_or_AI_generated',source_sha256=source_meta['source_sha256'],matting_manifest_sha256=sha(bundle/'matting-manifest.json'),source_crop_box=list(bbox),placement=[left,top],support_alpha_threshold=127,support_baseline=baseline,source_to_canvas_translation=[left-x1,top-y1],scale=1,rotation=0,opaque_core_pixels=int(core.sum()),opaque_core_rgb_max_error=error,source_pose=source_pose,material=material,framing=framing,quality='NOT_EVALUATED',fine_print='SOURCE_NATIVE_CORE_RETENTION_NOT_INDEPENDENT_TRANSCRIPTION',warnings=['Designed candidate, not polished-template acceptance or a recovered physical scene.','Source lighting is not changed; dark/colored scenes may have lighting mismatch.','Only alpha255pixels remain identical in the final RGB scene; edge pixels mix with the new background.','No upscaling, no camera rotation, no glass optics, no crop completion.','The soft floor shadow is a2Ddesign approximation, not recovered geometry.'])
    if regions_path:
        report['native_region_checks']=check_native_regions(rgba,result,[left-x1,top-y1],json.loads(regions_path.read_text()),source_meta['source_sha256'])
        report['regions_sha256']=sha(regions_path)
    report['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()};(out/'native-studio-manifest.json').write_text(json.dumps(report,indent=2));return report


def check_native_regions(rgba,scene,translation,regions,source_sha256):
    """Exact final-scene pixel check only for declared fully opaque source regions."""
    # Share the same rigor for shape/name/bounds validation without re-encoding
    # source bytes or pretending the PNG hash equals the original photo hash.
    h,w=rgba.shape[:2]
    if regions.get('source_sha256')!=source_sha256 or regions.get('image_size')!=[w,h]:raise ValueError('region source binding mismatch')
    items=regions.get('regions')
    if not isinstance(items,list) or not 1<=len(items)<=32:raise ValueError('one to32 regions required')
    dx,dy=translation;names=set();rows=[]
    for item in items:
        name=item.get('name');b=item.get('box')
        if not isinstance(name,str) or not name.strip() or name in names:raise ValueError('unique nonempty region names required')
        names.add(name)
        if not isinstance(b,list) or len(b)!=4 or any(type(x)!=int for x in b) or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('invalid source region box')
        x1,y1,x2,y2=b
        if not (0<=x1+dx<x2+dx<=scene.shape[1] and 0<=y1+dy<y2+dy<=scene.shape[0]):raise ValueError('region outside scene')
        src=rgba[y1:y2,x1:x2,:3];dest=scene[y1+dy:y2+dy,x1+dx:x2+dx];a=rgba[y1:y2,x1:x2,3];err=np.abs(src.astype('int16')-dest.astype('int16'))
        rows.append(dict(name=name,source_box=b,scene_box=[x1+dx,y1+dy,x2+dx,y2+dy],pixels=int(a.size),nonopaque_source_pixels=int((a!=255).sum()),final_rgb_changed_pixels=int(np.any(err,2).sum()),final_rgb_max_error=int(err.max()),source_native_retained=not bool(np.any(a!=255) or np.any(err))))
    return dict(all_source_native_regions_retained=all(r['source_native_retained'] for r in rows),regions=rows,transcription='NOT_PERFORMED',quality='NOT_EVALUATED')
