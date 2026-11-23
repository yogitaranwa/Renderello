"""Opt-in expanded unknown-band alpha/color recovery with source-bound core locks.
CPU research only. Derived trimap is a hypothesis, not semantic or optical truth.
Only the declared core stays source-exact; estimated edge RGB/alpha can change.
Use a guarded worker for untrusted/large jobs, as renderello.matting provides.
"""
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .matting import make_trimap,refine_arrays


def recover_expanded_edge(rgb,initial_alpha,protected,*,source_sha256,alpha_sha256,
        material,review_note,reviewed=False,radius=8):
    native_rgb(rgb);native_alpha(initial_alpha,rgb.shape[:2])
    if initial_alpha.size>1_000_000:raise ValueError('1MP source budget; no resize')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque source pairing review required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('source/core review note required')
    if source_sha256!=digest(rgb) or alpha_sha256!=digest(initial_alpha):
        raise ValueError('source/alpha binding mismatch')
    if not isinstance(protected,np.ndarray) or protected.dtype!=np.bool_ or protected.shape!=initial_alpha.shape or not protected.any():
        raise ValueError('nonempty native boolean protected core required')
    if np.any(initial_alpha[protected]!=255):raise ValueError('protected core must already be fully opaque')
    if type(radius) is not int or not 1<=radius<=12:raise ValueError('integer1..12band radius required')
    trimap=make_trimap(initial_alpha,radius)
    locked_from_unknown=int((protected&(trimap==128)).sum())
    if np.any(protected&(trimap==0)):raise ValueError('reviewed core contradicts definite background')
    trimap[protected]=255
    foreground,alpha,solver=refine_arrays(rgb.copy(),initial_alpha.copy(),trimap.copy(),method='cf',edge_colors=True)
    native_rgb(foreground);native_alpha(alpha,initial_alpha.shape)
    if foreground.shape!=rgb.shape:raise ValueError('solver foreground changed source coordinates')
    if not np.array_equal(foreground[trimap!=128],rgb[trimap!=128]):
        raise ValueError('solver changed definite foreground/background RGB')
    if not np.all(alpha[trimap==255]==255) or not np.all(alpha[trimap==0]==0):
        raise ValueError('solver violated trimap alpha locks')
    if not np.array_equal(foreground[protected],rgb[protected]):
        raise ValueError('protected source RGB changed')
    changed=np.any(foreground!=rgb,axis=2)
    return foreground,alpha,trimap,dict(schema='expanded-edge-recovery.v1',
        source_sha256=digest(rgb),initial_alpha_sha256=digest(initial_alpha),
        protected_sha256=digest(protected),foreground_sha256=digest(foreground),
        output_alpha_sha256=digest(alpha),trimap_sha256=digest(trimap),
        review_note=review_note,material=material,radius=radius,
        protected_pixels=int(protected.sum()),protected_rgb_exact=True,
        protected_relocked_from_unknown=locked_from_unknown,solver=solver,
        rgb_changed_pixels=int(changed.sum()),alpha_changed_pixels=int((alpha!=initial_alpha).sum()),
        initially_opaque_rgb_changed=int((changed&(initial_alpha==255)).sum()),
        source_resampled=False,fully_source_rgb_exact=not bool(changed.any()),
        optical_recovery=False,physical_relighting=False,independent_ground_truth=False,
        accepted=False,default_pipeline_changed=False,
        limitations=['Input alpha and trimap seeds may be wrong; a wider band can erase detail or include background.',
          'Foreground color estimation uses an internal pyramid. Native output dimensions do not mean every internal calculation is native resolution.',
          'Protected core is a reviewer declaration, not independent label verification.',
          'No geometry/pose/lighting/glass or cropped-part recovery.'])
