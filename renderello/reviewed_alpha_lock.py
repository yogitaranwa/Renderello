"""Opt-in reviewed alpha locks. Preserves a prior matte, not segmentation truth.

Opaque-source research only, no default pipeline integration. A reviewed native
region can retain its initial alpha including fractional boundary coverage.
The solver receives binary seeds in that region; fractional pixels are restored
explicitly after the solve. Such restoration can create seams and needs review.
Use prepare + guarded_file + finish for bounded real work. The array convenience
function is synchronous and deliberately does not promise a memory/time guard.
"""
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .matting import make_trimap,refine_arrays


def prepare_alpha_lock(rgb,initial_alpha,frozen,*,source_sha256,alpha_sha256,
        frozen_sha256,reviewed=False,material=None,review_note='',radius=8):
    native_rgb(rgb);native_alpha(initial_alpha,rgb.shape[:2])
    if initial_alpha.size>1_000_000:raise ValueError('1MP source budget; no resize')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque pairing/region review required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('nonempty review note up to2000characters required')
    if not isinstance(frozen,np.ndarray) or frozen.dtype!=np.bool_ or frozen.shape!=initial_alpha.shape or not frozen.any():
        raise ValueError('nonempty native boolean frozen region required')
    if source_sha256!=digest(rgb) or alpha_sha256!=digest(initial_alpha) or frozen_sha256!=digest(frozen):
        raise ValueError('source/alpha/region binding mismatch')
    if type(radius) is not int or not 1<=radius<=12:raise ValueError('integer1..12radius required')
    if not np.any(frozen&(initial_alpha>0)):raise ValueError('region must protect some prior foreground')
    tri=make_trimap(initial_alpha,radius)
    tri[frozen&(initial_alpha==0)]=0
    tri[frozen&(initial_alpha==255)]=255
    # Fractional prior coverage is not a binary foreground/background claim.
    tri[frozen&(initial_alpha>0)&(initial_alpha<255)]=128
    if int((tri==128).sum())>200_000:raise ValueError('200000unknown-pixel budget exceeded')
    return tri,dict(schema='reviewed-alpha-lock.v1',source_sha256=digest(rgb),
        initial_alpha_sha256=digest(initial_alpha),frozen_sha256=digest(frozen),
        trimap_sha256=digest(tri),review_note=review_note,material=material,radius=radius,
        frozen_pixels=int(frozen.sum()),frozen_foreground_pixels=int((frozen&(initial_alpha>0)).sum()),
        frozen_fractional_pixels=int((frozen&(initial_alpha>0)&(initial_alpha<255)).sum()),
        source_resampled=False,accepted=False,default_pipeline_changed=False,
        independent_ground_truth=False,optical_recovery=False,physical_relighting=False)


def finish_alpha_lock(rgb,initial_alpha,frozen,foreground,solved_alpha,trimap,**review):
    expected,meta=prepare_alpha_lock(rgb,initial_alpha,frozen,**review)
    native_rgb(foreground);native_alpha(solved_alpha,initial_alpha.shape)
    if foreground.shape!=rgb.shape or not np.array_equal(foreground,rgb):
        raise ValueError('alpha-only solver must preserve all source RGB')
    if not isinstance(trimap,np.ndarray) or trimap.dtype!=np.uint8 or not np.array_equal(trimap,expected):
        raise ValueError('solver trimap differs from reviewed bound trimap')
    if np.any(solved_alpha[expected==0]!=0) or np.any(solved_alpha[expected==255]!=255):
        raise ValueError('solver violated definite alpha seeds')
    result=solved_alpha.copy();result[frozen]=initial_alpha[frozen]
    meta.update(foreground_sha256=digest(foreground),solved_alpha_sha256=digest(solved_alpha),
        output_alpha_sha256=digest(result),all_source_rgb_exact=True,frozen_alpha_exact=True,
        frozen_pixels_restored_after_solve=int((frozen&(solved_alpha!=initial_alpha)).sum()),
        alpha_changed_pixels=int((result!=initial_alpha).sum()),
        limitations=['Prior alpha is preserved in a reviewer region, not recovered or verified as truth.',
          'Fractional restoration can create a transition seam; inspect the region perimeter.',
          'Outside the region the solver can still erase details or include background.',
          'Unchanged RGB does not prove visible product completeness or photographic quality.',
          'No pose, geometry, lighting, glass or missing-part reconstruction.'])
    return foreground.copy(),result,expected.copy(),meta


def recover_locked_alpha(rgb,initial_alpha,frozen,**review):
    tri,_=prepare_alpha_lock(rgb,initial_alpha,frozen,**review)
    foreground,alpha,solver=refine_arrays(rgb.copy(),initial_alpha.copy(),tri.copy(),method='cf',edge_colors=False)
    out=finish_alpha_lock(rgb,initial_alpha,frozen,foreground,alpha,tri,**review)
    out[3]['solver']=solver
    return out
