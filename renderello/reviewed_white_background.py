"""Source-bound outer-background extraction, not general subject segmentation.

The caller must inspect a high-contrast closed silhouette on a contiguous white
outer field first. Internal bright text is retained because only border-connected
pixels are removed. Output remains review-required, never automatically accepted.
"""
import hashlib
import numpy as np
import cv2


def outer_white_background_alpha(rgb, *, reviewed_source_sha256, minimum_channel=200,
                                 maximum_spread=35):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('native RGB uint8 required')
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=reviewed_source_sha256:
        raise ValueError('different source from explicitly reviewed outer-background case')
    if type(minimum_channel)is not int or not 180<=minimum_channel<=250:
        raise ValueError('explicit brightness threshold180..250 required')
    if type(maximum_spread)is not int or not 0<=maximum_spread<=50:
        raise ValueError('explicit neutral spread0..50 required')
    bright=(rgb.min(2)>minimum_channel)&(rgb.max(2).astype(int)-rgb.min(2).astype(int)<maximum_spread)
    n,labels=cv2.connectedComponents(bright.astype('uint8'),8)
    edge=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]));edge=edge[edge!=0]
    alpha=(~np.isin(labels,edge)*255).astype('uint8')
    if not (alpha==0).any()or not (alpha==255).any():raise ValueError('nontrivial outer background not established')
    return alpha,dict(source_sha256=reviewed_source_sha256,
        alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
        minimum_channel=minimum_channel,maximum_spread=maximum_spread,
        foreground_pixels=int((alpha==255).sum()),rgb_changed=False,
        semantic_completeness='NOT_AUTOMATICALLY_EVALUATED',accepted=False,
        limitations=['High-contrast closed silhouette on white outer field only.',
        'White foreground connected to the outer field can be deleted; review actual edges.',
        'Hard source-specific alpha, no antialias/decontamination or physical relighting.',
        'Matte review and photographic provenance are separate requirements.'])
