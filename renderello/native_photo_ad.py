"""Native whole-photo editorial ad layouts, separate from product staging.

No matting, scene generation, crop, resize, relighting or automatic approval.
"""
from dataclasses import dataclass,asdict
import hashlib
from io import BytesIO
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .candidate_diagnostics import native_rgb,digest

LAYOUTS={
 'square':dict(size=(1024,1024),well=(72,330,952,928),text=(76,88,948,294),footer=(76,970)),
 'portrait':dict(size=(1024,1280),well=(72,438,952,1036),text=(76,112,948,390),footer=(76,1160)),
 'landscape':dict(size=(1440,800),well=(528,80,1408,720),text=(48,132,480,650),footer=(48,742)),
}

@dataclass(frozen=True)
class AdCopy:
    eyebrow:str
    headline:str
    disclosure:str
    attribution:str
    def __post_init__(self):
        for n,limit in [('eyebrow',60),('headline',100),('disclosure',100),('attribution',120)]:
            value=getattr(self,n)
            if not isinstance(value,str)or not value.strip()or len(value)>limit:
                raise ValueError('bounded nonempty copy required')
            if any(ord(c)<32 and c!='\n'for c in value):raise ValueError('unsupported copy control character')
        if '\n'in self.eyebrow or '\n'in self.disclosure or '\n'in self.attribution or len(self.headline.split('\n'))>2:
            raise ValueError('one-line labels and at most two explicit headline lines required')


def render_native_ad(rgb,copy,*,source_sha256,layout,display_font,text_font,bold_font,palette):
    native_rgb(rgb)
    if rgb.shape[0]*rgb.shape[1]>2_000_000:raise ValueError('2MP native source cap')
    if source_sha256!=digest(rgb):raise ValueError('source pixel binding changed')
    if not isinstance(copy,AdCopy)or layout not in LAYOUTS:raise ValueError('typed copy and named layout required')
    if not isinstance(palette,dict)or set(palette)!={'ground','surface','ink','secondary','accent'}:
        raise ValueError('explicit five-color palette required')
    for value in palette.values():
        if not isinstance(value,str)or len(value)!=7 or value[0]!='#':raise ValueError('hex palette required')
        int(value[1:],16)
    spec=LAYOUTS[layout];x0,y0,x1,y1=spec['well'];h,w=rgb.shape[:2]
    if w>x1-x0 or h>y1-y0:raise ValueError('whole source does not fit photo well; no crop or resize')
    fonts={}
    for n,path in [('display',display_font),('text',text_font),('bold',bold_font)]:
        with open(path,'rb')as stream:fonts[n]=stream.read(5_000_001)
    if any(len(v)>5_000_000 for v in fonts.values()):raise ValueError('bounded font files required')
    canvas=Image.new('RGB',spec['size'],palette['ground']);draw=ImageDraw.Draw(canvas)
    tx,ty,tr,tb=spec['text'];eyefont=ImageFont.truetype(BytesIO(fonts['bold']),20)
    if draw.textbbox((0,0),copy.eyebrow,font=eyefont)[2]>tr-tx:raise ValueError('eyebrow does not fit; no truncation')
    draw.text((tx,ty),copy.eyebrow,font=eyefont,fill=palette['accent'])
    top=ty+48
    for size in range(100,47,-1):
        f=ImageFont.truetype(BytesIO(fonts['display']),size)
        box=draw.multiline_textbbox((0,0),copy.headline,font=f,spacing=4)
        if box[2]-box[0]<=tr-tx and box[3]-box[1]<=tb-top:break
    else:raise ValueError('headline does not fit; edit copy rather than silently shorten')
    draw.multiline_text((tx-box[0],top-box[1]),copy.headline,font=f,spacing=4,fill=palette['ink'])
    draw.rectangle(spec['well'],fill=palette['surface'])
    dx=x0+(x1-x0-w)//2;dy=y0+(y1-y0-h)//2
    canvas.paste(Image.fromarray(rgb),(dx,dy))
    footfont=ImageFont.truetype(BytesIO(fonts['text']),16);fx,fy=spec['footer']
    fb=draw.textbbox((0,0),copy.disclosure,font=footfont)
    if fb[2]-fb[0]>spec['size'][0]-fx-32:raise ValueError('disclosure does not fit')
    draw.text((fx,fy),copy.disclosure,font=footfont,fill=palette['secondary'])
    creditbox=draw.textbbox((0,0),copy.attribution,font=footfont)
    if creditbox[2]-creditbox[0]>spec['size'][0]-fx-32:raise ValueError('attribution does not fit')
    draw.text((fx,fy+24),copy.attribution,font=footfont,fill=palette['secondary'])
    out=np.array(canvas)
    if not np.array_equal(out[dy:dy+h,dx:dx+w],rgb):raise RuntimeError('whole native photograph changed')
    return out,dict(schema='native-photo-ad-layout.v1',layout=layout,source_sha256=digest(rgb),result_sha256=digest(out),
        copy=asdict(copy),font_sha256={n:hashlib.sha256(v).hexdigest()for n,v in fonts.items()},palette=palette,
        source_size=[w,h],canvas_size=list(spec['size']),integer_translation=[dx,dy],source_rgb_exact_pixels=w*h,
        source_cropped=False,source_resampled=False,source_relighting=False,matting_used=False,
        generated_scene=False,original_background_and_shadow_retained=True,headline_font_size=size,
        copy_is_draft_not_brand_or_owner_approved=True,photographic_acceptance=False,polished_template_acceptance=False,
        category='GRAPHIC_DESIGN_CONCEPT_NOT_NEW_PRODUCT_PHOTOGRAPHY',
        limitations=['Original source photo, lighting, background and shadow remain unchanged.',
          'Graphic copy is draft input, not approved brand messaging or verified product claims.',
          'Three layouts of the same source are not three source photos or distinct template categories.',
          'No source rights clearance, photoshoot acceptance or production default is implied.'])
