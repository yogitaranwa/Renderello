"""Separate feathered attenuation of reviewed distant background, not segmentation.
No zero-mask seam override: the selection boundary and a clearance collar are identity.
Real low-alpha detail can be erased by a wrong annotation. No default integration.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_dilation
from .candidate_diagnostics import native_rgb,native_alpha,digest

def attenuate_reviewed_background(rgb,alpha,background,protected,*,source_sha256,
 alpha_sha256,background_sha256,protected_sha256,reviewed=False,material=None,
 review_note='',ceiling=32,clearance=12,feather=32):
 native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
 if alpha.size>1_000_000:raise ValueError('1MP native cap; no resize')
 if reviewed is not True or material!='opaque':raise ValueError('explicit opaque background review required')
 if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:raise ValueError('bounded review note required')
 for m in (background,protected):
  if not isinstance(m,np.ndarray) or m.dtype!=np.bool_ or m.shape!=alpha.shape or not m.any():raise ValueError('nonempty native boolean masks required')
 for arr,h in [(rgb,source_sha256),(alpha,alpha_sha256),(background,background_sha256),(protected,protected_sha256)]:
  if digest(arr)!=h:raise ValueError('source/alpha/mask binding mismatch')
 for v,lo,hi in [(ceiling,1,32),(clearance,4,64),(feather,16,64)]:
  if type(v)is not int or not lo<=v<=hi:raise ValueError('bounded integer controls required')
 if np.any(background&protected) or not np.any(protected&(alpha>0)):raise ValueError('disjoint background and retained foreground required')
 if np.any(alpha[background]>ceiling):raise ValueError('selected background exceeds alpha ceiling; no partial selection')
 anchors=(alpha>=128)|protected
 if not np.any(alpha>=128):raise ValueError('high-alpha foreground anchor required')
 # EDT to outside the selected region, plus an explicit frame collar. The frame
 # is identity too; this avoids inventing an unobserved beyond-frame selection.
 padded=np.pad(background,1,constant_values=False)
 selection_distance=distance_transform_edt(padded)[1:-1,1:-1]
 anchor_distance=distance_transform_edt(~anchors)
 t=np.clip((np.minimum(selection_distance,anchor_distance)-clearance)/feather,0,1)
 weight=t*t*(3-2*t);weight[~background]=0
 out=np.rint(alpha.astype(float)*(1-weight)).astype(np.uint8)
 collar=background&(np.minimum(selection_distance,anchor_distance)<=clearance)
 if not np.array_equal(out[collar],alpha[collar]) or not np.array_equal(out[~background],alpha[~background]) or not np.array_equal(out[protected],alpha[protected]) or np.any(out>alpha):raise RuntimeError('attenuation identity/monotonicity invariant failed')
 changed=out!=alpha;seam=background&binary_dilation(~background,structure=np.ones((3,3)))
 if np.any(changed&seam):raise RuntimeError('selection boundary changed')
 return rgb.copy(),out,dict(schema='reviewed-background-attenuation.v1',source_sha256=digest(rgb),alpha_sha256=digest(alpha),background_sha256=digest(background),protected_sha256=digest(protected),output_alpha_sha256=digest(out),review_note=review_note,ceiling=ceiling,clearance=clearance,feather=feather,alpha_changed_pixels=int(changed.sum()),maximum_removed_alpha=int((alpha-out).max()),removed_alpha_units=int((alpha.astype(np.int64)-out).sum()),selection_boundary_alpha_exact=True,selection_and_frame_collar_exact=True,protected_alpha_exact=True,outside_selection_alpha_exact=True,source_rgb_exact=True,source_resampled=False,accepted=False,default_changed=False,physical_recovery=False,limitations=['Reviewer background annotation is not semantic truth; real low-alpha detail can be erased.','A broad feather can retain visible background haze near the protected enclosure or frame.','This does not repair source edge color, geometry, lighting or glass transmission.','No photographic acceptance or default promotion.'])
