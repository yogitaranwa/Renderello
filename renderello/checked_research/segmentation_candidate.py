"""Import a reviewed source-RGB-exact BEN2 candidate, never automatic acceptance.
Model inference resizes to1024square and returns resampled native-sized alpha.
This wrapper validates source alignment by byte comparison, not semantic truth.
No model download, inference, mask cleanup or alpha thresholding occurs here.
"""
import numpy as np

def import_rgb_exact_candidate(source, candidate, *, pairing_reviewed=False, internal_resampling_declared=False):
    if pairing_reviewed is not True:
        raise ValueError('explicit source/candidate pairing review required')
    if internal_resampling_declared is not True:
        raise ValueError('internal model resize and returned-alpha resampling must be declared')
    if not isinstance(source,np.ndarray) or source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3:
        raise ValueError('native uint8 RGB source required')
    h,w=source.shape[:2]
    if min(h,w)<2 or h*w>4_194_304:
        raise ValueError('native source2pixelsminimum peraxis,4MPmaximum')
    if not isinstance(candidate,np.ndarray) or candidate.dtype!=np.uint8 or candidate.shape!=(h,w,4):
        raise ValueError('matching native-sized uint8 RGBA candidate required')
    if not np.array_equal(source,candidate[:,:,:3]):
        raise ValueError('candidate RGB differs from source; foreground-refined output is not source-locked')
    a=candidate[:,:,3]
    if not (a>127).any() or not (a==0).any():
        raise ValueError('candidate needs nontrivial foreground and zero-alpha background')
    return candidate.copy(),dict(source_rgb_exact=True,source_resampled_by_wrapper=False,
        native_model_inference=False,internal_model_size=[1024,1024],alpha_resampling_declared=True,
        opaque_pixels=int((a==255).sum()),fractional_pixels=int(((a>0)&(a<255)).sum()),
        zero_pixels=int((a==0).sum()),foreground_pixels=int((a>127).sum()),
        alpha_accuracy=None,publication_accepted=False,production_promoted=False,
        pairing_review_is_caller_declaration=True)
