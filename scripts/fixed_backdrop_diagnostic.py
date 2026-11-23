"""Offline native fixed-backdrop control. No inference/network/paid backend.
Explicit source, alpha and inspected background required. No physical lighting claim.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from renderello.compose import contact_shadow
from renderello.cutout import _bbox_from_matte
from renderello.native_pixel_audit import audit_native_pixels

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['source','alpha','background','out']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--background-provenance',required=True)
 p.add_argument('--source-origin',nargs=2,type=int,required=True,metavar=('X','Y'))
 p.add_argument('--no-shadow',action='store_true');a=p.parse_args()
 if a.out.exists():raise ValueError('Preserve previous output; choose a fresh directory')
 s=np.array(Image.open(a.source).convert('RGB'));alpha=np.array(Image.open(a.alpha))
 bg=np.array(Image.open(a.background).convert('RGB'))
 if alpha.dtype!=np.uint8 or alpha.shape!=s.shape[:2]:raise ValueError('Source-sized uint8 alpha required')
 if not np.any(alpha==255):raise ValueError('No fully opaque pixels to audit')
 x,y,w,h=_bbox_from_matte(alpha);px=a.source_origin[0]+x;py=a.source_origin[1]+y
 ch,cw=bg.shape[:2]
 if min(px,py)<0 or px+w>cw or py+h>ch:raise ValueError('Native product does not fit; resizing refused')
 al=alpha[y:y+h,x:x+w];plate=bg.astype(np.float32)
 if not a.no_shadow:plate*=1-contact_shadow((cw,ch),(px,py),(w,h),al,'soft faint contact shadow directly beneath product')[:,:,None]/255*.85
 f=al[:,:,None].astype(np.float32)/255;region=plate[py:py+h,px:px+w];region[:]=region*(1-f)+s[y:y+h,x:x+w]*f
 out=np.clip(plate,0,255).astype(np.uint8);audit=audit_native_pixels(s,alpha,out,[x,y,w,h],[px,py,w,h])
 if audit['status']!='MATCH':raise ValueError('Native opaque pixel audit failed')
 a.out.mkdir(parents=True);Image.fromarray(out).save(a.out/'fixed-background-composite.png')
 sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
 record=dict(background_provenance=a.background_provenance,source_sha256=sha(a.source),alpha_sha256=sha(a.alpha),background_sha256=sha(a.background),output_sha256=sha(a.out/'fixed-background-composite.png'),source_origin=a.source_origin,native_pixel_audit=audit,source_resized=False,new_generation=False,provider_calls=0,contact='none' if a.no_shadow else 'current_default_2D_contact_cue_NOT_inferred_geometry',human_review='PENDING',quality_acceptance='NOT_ESTABLISHED',mask_semantics='NOT_VERIFIED')
 (a.out/'record.json').write_text(json.dumps(record,indent=2));print(a.out/'record.json')
if __name__=='__main__':main()
