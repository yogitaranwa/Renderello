"""Reviewed finite surface-shadow supplement. Designed2D light cue, not relighting.
This never estimates geometry, camera or source illumination. An explicit source-
bound review is required. It only changes the background below the product.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear,srgb


@dataclass(frozen=True)
class SurfaceShadow:
    center_x:int
    floor_row:int
    sigma_x:float
    sigma_y:float
    opacity:float
    source_sha256:str
    alpha_sha256:str
    background_sha256:str
    contact_sha256:str
    review_note:str
    reviewed:bool=False
    surface_kind:str='continuous_neutral_floor'

    def __post_init__(self):
        if self.reviewed is not True or self.surface_kind!='continuous_neutral_floor':
            raise ValueError('reviewed continuous neutral floor required; no graphic wall/plinth assumption')
        if any(type(v) is not int or v<0 for v in [self.center_x,self.floor_row]):raise ValueError('native integer anchor required')
        for key,low,high in [('sigma_x',2,256),('sigma_y',1,32),('opacity',.01,.35)]:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not low<=v<=high:
                raise ValueError('bounded finite '+key+' required')
        for key in ['source_sha256','alpha_sha256','background_sha256','contact_sha256']:
            v=getattr(self,key)
            if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('pixel hashes required')
        if not isinstance(self.review_note,str) or not self.review_note.strip() or len(self.review_note)>2000:raise ValueError('source light and surface review note required')


def supplement_surface_shadow(rgb,alpha,background,contact,policy,*,fractional_underlay=False):
    if type(fractional_underlay) is not bool:raise ValueError('explicit boolean fractional underlay mode required')
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2]);native_rgb(background)
    if background.shape!=rgb.shape:raise ValueError('same native canvas required')
    if not isinstance(policy,SurfaceShadow):raise ValueError('typed source-bound surface review required')
    if not isinstance(contact,np.ndarray) or contact.dtype.kind!='f' or contact.shape!=alpha.shape or not np.isfinite(contact).all() or np.any((contact<0)|(contact>.5)):
        raise ValueError('finite floating contact shade0..0.5 required')
    for a,key in [(rgb,'source_sha256'),(alpha,'alpha_sha256'),(background,'background_sha256'),(contact,'contact_sha256')]:
        if digest(a)!=getattr(policy,key):raise ValueError('review belongs to different '+key)
    h,w=alpha.shape
    if policy.center_x>=w or policy.floor_row>=h-2:raise ValueError('floor anchor outside canvas')
    hard=alpha>=128
    if not hard.any() or hard.all() or np.any(contact[hard]):raise ValueError('nontrivial hard support with shadow excluded required')
    rows=np.flatnonzero(hard[:,policy.center_x])
    if not len(rows) or abs(int(rows[-1])-policy.floor_row)>2:
        raise ValueError('floor center must bind to the actual native base within2pixels')
    if contact[int(rows[-1])+1,policy.center_x]<=0:raise ValueError('existing contact missing at reviewed base')
    # Symmetric floor footprint avoids a hard horizontal onset at the floor row.
    # Hard foreground excludes the shadow; background beside/behind the base can shade.
    # Finite3sigma support is an illustration bound, not inferred surface geometry.
    y,x=np.indices(alpha.shape);xx=(x-policy.center_x)/policy.sigma_x;yy=(y-policy.floor_row-1)/policy.sigma_y
    broad=policy.opacity*np.exp(-.5*(xx*xx+yy*yy))
    # The opt-in mode shades the background under fractional coverage, rather
    # than treating alpha128..254 as fully opaque. Original default retained.
    exclusion=(alpha==255) if fractional_underlay else hard
    broad[(abs(xx)>3)|(abs(yy)>3)|exclusion]=0
    shade=1-(1-contact)*(1-broad);shade[exclusion]=0
    result=linear_over(rgb,alpha,srgb(linear(background)*(1-shade[...,None])))
    if not np.array_equal(result[alpha==255],rgb[alpha==255]):raise RuntimeError('opaque core changed')
    return result,shade,dict(schema='reviewed-surface-shadow.v1',policy=asdict(policy),
        output_sha256=digest(result),shadow_sha256=digest(shade),
        broad_pixels=int((broad>0).sum()),fractional_underlay=fractional_underlay,
        shadow_exclusion='alpha255_only' if fractional_underlay else 'alpha128_and_above_legacy',
        fractional_underlay_pixels=int(((shade>0)&(alpha>0)&(alpha<255)).sum()),
        source_rgb_repainted=False,source_resampled=False,
        source_lighting_unchanged=True,physical_relighting=False,recovered_3d_geometry=False,
        opaque_input_rgb_exact=True,accepted=False,independent_review=False,
        limitation='Manually designed finite floor attenuation. Review cannot establish actual source lighting, geometry or photographic acceptance.')
