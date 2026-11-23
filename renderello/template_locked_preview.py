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


def compose_template_contact(source,alpha,output,*,protected_regions,offset,canvas=(1024,1024),background=None):
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
    if background is not None:
        if background.size!=canvas or background.mode!='RGB':raise ValueError('background must match RGB canvas')
        bg=np.asarray(background).astype(float)
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


def preview_reviewed(source,alpha_path,regions,root,*,reviewed=False,template='minimal_tabletop',instruction='',alternatives=2,material='unknown',offset=None,aspect=None,scene_review=None):
    from .colab_demo import validate_choices
    from .compose import procedural_background
    import shutil
    spec=validate_choices(template,instruction,alternatives)
    from .schemas import Aspect
    # Preserve the legacy square default; template framing is explicit opt-in.
    chosen_aspect=Aspect.SQUARE if aspect is None else (spec.aspect if aspect=='template' else Aspect(aspect))
    canvas=chosen_aspect.size
    if reviewed is not True:raise ValueError('Explicit alpha review required')
    if material not in ('unknown','opaque','transparent'):raise ValueError('Unknown material')
    if material=='transparent':raise ValueError('Reviewed RGB paste-back is not optical transparency recovery')
    source,alpha_path,root=Path(source),Path(alpha_path),Path(root)
    if root.exists():raise ValueError('Preserve earlier results')
    im=Image.open(source).convert('RGB');alpha=Image.open(alpha_path).convert('L')
    if alpha.size!=im.size:raise ValueError('Native source/alpha dimensions differ')
    if not isinstance(regions,dict) or not regions:raise ValueError('Protected regions required')
    for name,box in regions.items():
        if not isinstance(name,str) or not name or len(box)!=4 or any(type(v)!=int for v in box):raise ValueError('Invalid protected region')
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=im.width and 0<=y1<y2<=im.height):raise ValueError('Protected region outside source')
        if np.any(np.asarray(alpha)[y1:y2,x1:x2]!=255):raise ValueError('Protected region must be opaque')
    compatibility=None
    if scene_review is not None:
        from .template_compatibility import plan_template
        compatibility=plan_template(np.array(im),np.array(alpha),template,scene_review,aspect=chosen_aspect)
        if not compatibility['eligible_for_reviewed_preview']:
            raise ValueError('Source/template compatibility: '+', '.join(compatibility['blockers']))
    bb=alpha.getbbox()
    if not bb:raise ValueError('Empty alpha')
    if offset is None:
        offset=compatibility['whole_source_translation'] if compatibility is not None else [(canvas[0]-(bb[2]-bb[0]))//2-bb[0],(canvas[1]-(bb[3]-bb[1]))//2-bb[1]]
    if len(offset)!=2 or any(type(v)!=int for v in offset):raise ValueError('integer translation required')
    if compatibility is not None:
        if min(offset)<0 or offset[0]+im.width>canvas[0] or offset[1]+im.height>canvas[1]:
            raise ValueError('Compatibility plan retains whole native source; requested placement clips source frame')
        compatibility['executed_source_translation']=list(offset)
    if bb[0]+offset[0]<0 or bb[1]+offset[1]<0 or bb[2]+offset[0]>canvas[0] or bb[3]+offset[1]>canvas[1]:
        raise ValueError('native reviewed source does not fit requested aspect; no silent resize or crop')
    root.mkdir();shutil.copyfile(source,root/('source-original'+source.suffix));alpha.save(root/'reviewed-alpha.png');outputs=[];audits=[]
    for i in range(alternatives):
        bg=Image.fromarray(procedural_background(canvas,spec,i)).convert('RGB')
        out=root/('alternative-'+str(i+1));audit=compose_template_contact(im,alpha,out,protected_regions=regions,offset=list(offset),canvas=canvas,background=bg);outputs.append(str(out/'output.png'));audits.append(audit)
    manifest={'kind':'REVIEWED_SOURCE_LOCK_TEMPLATE_CPU_CONTROL_NOT_PRODUCTION_ACCEPTANCE','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'template':template,'aspect':chosen_aspect.value,'canvas':list(canvas),'template_aspect':spec.aspect.value,'aspect_selection':'legacy_square_default' if aspect is None else 'explicit','custom_instruction_recorded_not_generatively_executed':instruction,'manual_alpha_review_required':True,'transparent_optics_recovered':False,'audits':audits,'outputs':outputs,'paid_credits':0,'source_template_compatibility':compatibility}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'kind':manifest['kind'],'outputs':outputs,'manifest':str(root/'manifest.json'),'root':str(root)}
