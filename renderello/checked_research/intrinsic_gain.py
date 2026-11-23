"""Opt-in, source-detail-preserving intrinsic gain experiment, not physical relighting.
Only a bounded achromatic gain is applied to the original linear RGB. Estimated
model RGB never replaces labels. This can still damage brightness/materials.
Caller must review source/map alignment. No registration inference or source resize.
"""
import numbers
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from .detail_transfer import decode, encode

def intrinsic_gain(source, alpha, prediction, *, alignment_reviewed=False, sigma=16., shading_strength=.5, residual_suppression=.25, gain_bounds=(.65,1.45)):
 if alignment_reviewed is not True:raise ValueError('reviewed source/map alignment required')
 if not isinstance(source,np.ndarray) or source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3:raise ValueError('uint8 RGB required')
 h,w=source.shape[:2]
 if min(h,w)<2 or h*w>4_194_304:raise ValueError('source must be2pixels minimum per axis and4MP maximum')
 if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=(h,w) or not (alpha==255).any():raise ValueError('matching uint8 alpha with opaque foreground required')
 if not isinstance(prediction,np.ndarray) or prediction.shape!=(3,h,w,3) or prediction.dtype.kind!='f' or not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1:raise ValueError('finite linear3target RGB maps in0to1required')
 for v,low,high in [(sigma,0,100),(shading_strength,0,1),(residual_suppression,0,1)]:
  if isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) or not np.isfinite(v) or not low<=v<=high:raise ValueError('invalid scalar control')
 if sigma==0:raise ValueError('positive sigma required')
 try:lo,hi=gain_bounds
 except (TypeError,ValueError):raise ValueError('gain bounds pair required') from None
 if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) for v in [lo,hi]) or not np.isfinite([lo,hi]).all() or not 0<lo<=1<=hi<=2:raise ValueError('gain bounds must straddle1,max2')
 s=decode(source);weights=np.array([.2126,.7152,.0722]);y=(s*weights).sum(2);a,sh,r=prediction.astype(np.float64);opaque=alpha==255;support=alpha>0;mask=alpha/255.;den=gaussian_filter(mask,sigma)
 def blur(v):return gaussian_filter(v*mask,sigma)/np.maximum(den,1e-8)
 shade=blur((sh*weights).sum(2));res=blur((r*weights).sum(2));base=(a*sh*weights).sum(2)
 scales,_=nnls(np.stack([base[opaque],(r*weights).sum(2)[opaque]],axis=1),y[opaque]);median=float(np.median(shade[opaque]));identity=shading_strength==0 and residual_suppression==0
 if not identity and median<=1e-8:raise ValueError('unusable near-zero model shading')
 gain=(np.maximum(median,1e-8)/np.maximum(shade,1e-8))**shading_strength
 ratio=np.minimum(scales[1]*res/np.maximum(blur(y),1e-8),1.)
 gain=np.clip(gain*(1-residual_suppression*ratio),lo,hi);requested=gain.copy();cap=1/np.maximum(s.max(2),1e-12);gain=np.minimum(gain,cap);gain[~support]=1
 rgb=encode(s*gain[:,:,None]);rgb[~support]=source[~support]
 if identity:rgb=source.copy();gain=np.ones((h,w),float)
 rgba=np.dstack([rgb,alpha]);return rgba,gain,dict(method='bounded original linear-RGB gain from blurred intrinsic shading/residual; in-sample scale fit',source_resampled=False,source_rgb_exact=identity,estimated_rgb_replaces_source=False,physical_relighting=False,publication_accepted=False,independent_accuracy=None,gain_bounds=[lo,hi],sigma=float(sigma),shading_strength=float(shading_strength),residual_suppression=float(residual_suppression),fitted_luminance_scales=scales.tolist(),highlight_capped_pixels=int(((requested>cap)&support).sum()),preencode_out_of_gamut_pixels=int(((s*gain[:,:,None]>1+1e-12).any(2)&support).sum()),opaque_rgb_changed=int((np.any(rgb!=source,2)&opaque).sum()))
