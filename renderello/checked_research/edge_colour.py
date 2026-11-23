"""Experimental fractional-edge colour estimation, not exact product recovery.

The input RGB is retained at alpha 0 and 255. Fractional edges may change RGB.
Native shape is preserved, but the estimator uses an internal image pyramid.
"""
import numpy as np


def estimate_fractional_edge(rgb, alpha, *, source_alpha_pair_reviewed=False, estimator=None):
    if source_alpha_pair_reviewed is not True:
        raise ValueError('explicit source/alpha pairing review required')
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('native uint8 RGB required')
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=rgb.shape[:2]:
        raise ValueError('matching native uint8 alpha required')
    if min(alpha.shape)<2 or alpha.size>1_000_000:
        raise ValueError('native dimensions2..1MP required')
    fractional=(alpha>0)&(alpha<255)
    if not (alpha==0).any() or not (alpha==255).any() or not fractional.any():
        raise ValueError('known foreground/background and fractional edge required')
    injected=estimator is not None
    if estimator is None:
        from pymatting import estimate_foreground_ml
        estimator=estimate_foreground_ml
    a=alpha.astype(np.float32)/255
    foreground,background=estimator(rgb.astype(np.float32)/255,a,return_background=True)
    for value in [foreground,background]:
        if not isinstance(value,np.ndarray) or value.shape!=rgb.shape or value.dtype.kind!='f' or not np.isfinite(value).all() or ((value<0)|(value>1)).any():
            raise ValueError('finite native floating foreground/background in[0,1] required')
    corrected=rgb.copy()
    corrected[fractional]=np.round(foreground[fractional]*255).astype(np.uint8)
    metrics=dict(method='multilevel foreground/background estimate at fractional alpha only',
                 fractional_pixels=int(fractional.sum()),rgb_changed_pixels=int(np.any(corrected!=rgb,2).sum()),
                 opaque_rgb_exact=True,zero_alpha_rgb_exact=True,alpha_changed=False,
                 native_output_shape=True,estimator_uses_internal_pyramid=True,
                 source_rgb_fully_exact=False,physical_material_recovery=False,
                 pairing_review_is_caller_declaration=True,injected_estimator=injected,
                 publication_accepted=False,boundary_accuracy=None)
    return corrected,metrics
