"""Experimental aligned-frame luminance transfer, not physical relighting.
Inputs require an independently checked registration. Shape checks cannot prove it.
No source resampling or generated RGB replacement. This deliberately changes RGB.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

def decode(x):
 x=x.astype(float)/255
 return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def encode(x):
 return np.round(np.clip(np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055),0,1)*255).astype('uint8')

def transfer(source,raw,alpha,*,registration_reviewed=False,sigma=16.,gain_bounds=(.65,1.45)):
 if registration_reviewed is not True:raise ValueError('registration review required; array shape is not alignment evidence')
 if source.dtype!=np.uint8 or raw.dtype!=np.uint8 or source.shape!=raw.shape or source.ndim!=3 or source.shape[2]!=3:raise ValueError('matching uint8 RGB source/raw required')
 h,w=source.shape[:2]
 if h*w>4_194_304:raise ValueError('research limit4MP')
 if alpha.shape!=(h,w) or alpha.dtype!=np.uint8:raise ValueError('matching uint8 alpha required')
 if not np.any(alpha):raise ValueError('empty alpha')
 if not np.isfinite(sigma) or sigma<=0 or sigma>100:raise ValueError('sigma out of bounds')
 lo,hi=gain_bounds
 if not np.isfinite([lo,hi]).all() or not (0<lo<=1<=hi<=2):raise ValueError('gain bounds must straddle1, max2')
 a=alpha/255.;s=decode(source);r=decode(raw);y=np.array([.2126,.7152,.0722]);den=gaussian_filter(a,sigma);sl=gaussian_filter((s*y).sum(2)*a,sigma)/np.maximum(den,1e-6);dl=gaussian_filter((r*y).sum(2)*a,sigma)/np.maximum(den,1e-6)
 gain=np.clip((dl+1e-8)/(sl+1e-8),lo,hi);requested=gain.copy();cap=1/np.maximum(s.max(2),1e-12);gain=np.minimum(gain,cap);gain[a==0]=1;fg=encode(s*gain[:,:,None]);out=np.round(fg*a[:,:,None]+raw*(1-a[:,:,None])).astype('uint8')
 return out,gain,dict(method='alpha-normalized linear-luminance Gaussian ratio; per-pixel no-new-clipping cap',source_rgb_exact=False,source_resampled=False,physical_relighting=False,publication_accepted=False,highlight_capped_pixels=int(((requested>cap)&(a>0)).sum()),preencode_out_of_gamut_pixels=int(((s*gain[:,:,None]>1+1e-12).any(2)&(a>0)).sum()),opaque_rgb_changed=int((np.any(out!=source,2)&(alpha==255)).sum()))
