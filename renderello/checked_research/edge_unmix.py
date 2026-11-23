"""Opt-in local fractional-edge unmixing. Not true foreground recovery.
Nearest known background/interior RGB are assumptions, not observations under edge.
All alpha, alpha0/255RGB and coordinates unchanged. Linear/sRGB mixing selectable
only by an explicit caller choice; source camera compositing space is unknown.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

def unmix_edge(rgb,alpha,*,pairing_reviewed=False,mixing_space=None,regularization=.05,max_distance=12.):
 if pairing_reviewed is not True:raise ValueError('source/alpha pairing review required')
 if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('uint8 RGB required')
 if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=rgb.shape[:2]:raise ValueError('matching uint8alpha required')
 if min(alpha.shape)<2 or alpha.size>1_000_000:raise ValueError('2pixel axes/1MP limit')
 if mixing_space not in ['linear','srgb']:raise ValueError('explicit linear/srgb mixing space required')
 for v in [regularization,max_distance]:
  if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v):raise ValueError('finite scalar controls required')
 if not 0<regularization<=1 or not 0<max_distance<=100:raise ValueError('control bounds')
 f=alpha==255;b=alpha==0;edge=(alpha>0)&(alpha<255)
 if not f.any() or not b.any() or not edge.any():raise ValueError('known foreground/background/fractionalpixels required')
 db,ib=distance_transform_edt(~b,return_indices=True);df,iff=distance_transform_edt(~f,return_indices=True)
 x=rgb.astype(float)/255
 if mixing_space=='linear':x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 prior=x[iff[0],iff[1]];background=x[ib[0],ib[1]];a=alpha[:,:,None].astype(float)/255
 estimate=(a*(x-(1-a)*background)+regularization*prior)/(a*a+regularization)
 eligible=edge&(db<=max_distance)&(df<=max_distance);clipped=((estimate<0)|(estimate>1)).any(2)&eligible
 estimate=np.clip(estimate,0,1)
 if mixing_space=='linear':estimate=np.where(estimate<=.0031308,12.92*estimate,1.055*estimate**(1/2.4)-.055)
 out=rgb.copy();out[eligible]=np.round(estimate[eligible]*255).astype('uint8')
 return out,dict(method='regularized nearest-known local edge unmixing',mixing_space=mixing_space,regularization=float(regularization),max_distance=float(max_distance),fractional_pixels=int(edge.sum()),eligible_pixels=int(eligible.sum()),distance_rejected_pixels=int((edge&~eligible).sum()),changed_pixels=int(np.any(out!=rgb,2).sum()),preclip_estimate_out_of_range_pixels=int(clipped.sum()),opaque_rgb_exact=True,zero_alpha_rgb_exact=True,alpha_changed=False,source_resampled=False,physical_recovery=False,boundary_accuracy=None,publication_accepted=False)
