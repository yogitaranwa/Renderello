"""Free boundary/contact control. Native source RGB and reviewed label cores stay locked.
Not a physical renderer, automatic mask reviewer, or transparent-material recovery.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,gaussian_filter


def refine_alpha(alpha, protected_regions, *, trim=0.7, feather=1.4, contour=None):
    a=np.asarray(alpha.convert('L'))
    if trim<0 or feather<=0:raise ValueError('invalid boundary width')
    mask=a>0
    if contour is not None:
        c=Image.new('L',alpha.size);ImageDraw.Draw(c).polygon(contour,fill=255)
        mask &= np.array(c)>0
    dist=distance_transform_edt(mask)
    result=np.minimum(a,np.clip((dist-trim)/feather,0,1)*255).astype('uint8')
    for box in protected_regions.values():
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=a.shape[1] and 0<=y1<y2<=a.shape[0]):raise ValueError('invalid protected box')
        if np.any(a[y1:y2,x1:x2]!=255):raise ValueError('protected region not opaque')
        if np.any(~mask[y1:y2,x1:x2]):raise ValueError('contour crosses protected region')
        result[y1:y2,x1:x2]=255
    return Image.fromarray(result)


def compose_contact(source,alpha,output,*,protected_regions,offset,canvas=(1024,1024)):
    output=Path(output)
    if output.exists():raise ValueError('preserve previous output')
    rgb=np.array(source.convert('RGB'));a=np.array(alpha.convert('L'))
    if rgb.shape[:2]!=a.shape or rgb.shape[0]*rgb.shape[1]>4000000:raise ValueError('invalid source/mask')
    if len(offset)!=2 or any(type(v)!=int for v in offset):raise ValueError('integer translation required')
    if not protected_regions:raise ValueError('reviewed label/texture regions required')
    dx,dy=offset;h,w=canvas[1],canvas[0];ys,xs=np.nonzero(a)
    if not len(xs) or min(xs)+dx<0 or min(ys)+dy<0 or max(xs)+dx>=w or max(ys)+dy>=h:raise ValueError('empty or clipped source')
    yy,xx=np.mgrid[:h,:w];grad=np.clip((yy/h)*8+((xx-w*.4)/w)**2*8,0,15)
    bg=np.stack([242-grad,239-grad,234-grad],axis=-1)
    # Support comes from the lowest reviewed silhouette, not its box centre.
    opaque=a>=128;bottom=np.full(a.shape[1],-1)
    for x in np.where(opaque.any(axis=0))[0]:bottom[x]=np.where(opaque[:,x])[0][-1]
    low=int(bottom.max());support_x=np.where(bottom>=low-5)[0]
    cx=float(support_x.mean()+dx);cy=low+dy
    support_width=max(float(np.ptp(support_x)),12)
    # A compact contact term plus a soft directional tabletop shadow.
    near=np.exp(-((xx-cx)/(support_width*.60+5))**2-((yy-cy-1)/3.8)**2)*.28
    broad=np.exp(-((xx-cx-15)/(max(40,support_width*.85)))**2-((yy-cy-5)/12)**2)*.12
    shade=np.clip(gaussian_filter(near,1)+gaussian_filter(broad,4),0,.40)
    bg=bg*(1-shade[...,None])+np.array([70,65,60])*shade[...,None]
    base=Image.fromarray(np.round(bg).astype('uint8'));base.paste(source.convert('RGB'),(dx,dy),alpha)
    seen=np.array(base);checks=[]
    for name,box in protected_regions.items():
        x1,y1,x2,y2=box
        if np.any(a[y1:y2,x1:x2]!=255):raise ValueError('protected region must be opaque')
        err=np.abs(seen[y1+dy:y2+dy,x1+dx:x2+dx].astype(int)-rgb[y1:y2,x1:x2].astype(int))
        if np.any(err):raise RuntimeError('source label changed')
        checks.append({'name':name,'source_box':box,'pixels':int((x2-x1)*(y2-y1)),'changed_pixels':0,'max_channel_error':0})
    oy,ox=np.nonzero(a==255);diff=seen[oy+dy,ox+dx].astype(int)-rgb[oy,ox].astype(int)
    assert not np.any(diff)
    output.mkdir(parents=True);base.save(output/'output.png');alpha.save(output/'reviewed-alpha.png');source.save(output/'source-native.png')
    manifest={'kind':'BOUNDARY_CONTACT_CONTROL_NOT_PRODUCTION_APPROVAL','integer_translation':[dx,dy],'source_resampled':False,'source_rgb_repainted':False,'transparent_lid_recovered':False,'support_centre':[cx,int(cy)],'support_width':support_width,'protected_regions':checks,'opaque_pixels':len(ox),'opaque_changed_pixels':0,'output_sha256':hashlib.sha256((output/'output.png').read_bytes()).hexdigest(),'limitations':['Reviewed alpha boundary may differ from original alpha.','Contact shadow is a 2D support heuristic, not recovered scene geometry.','Transparent lid RGB retains old-scene transmission and reflections. No measured clean plate exists.','Unchanged source perspective may remain physically unsupported.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
