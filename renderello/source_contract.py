"""Explicit source-camera contract and non-semantic mask preflight.

Never infer pose, product identity, transparency or label readability from a filename.
Declarations are reviewer inputs, not automatic vision or authenticated approval.
"""
from dataclasses import asdict, dataclass
import hashlib
import cv2
import numpy as np
from .boundary import frame_edge_contact

POSES=('unknown','front','elevated','overhead','suspended')


@dataclass(frozen=True)
class SourceContract:
    pose: str = 'unknown'
    material: str = 'unknown'
    framing: str = 'unknown'
    genuine_tiny_text: bool = False

    def __post_init__(self):
        if self.pose not in POSES:raise ValueError('unsupported source pose')
        if self.material not in ('unknown','opaque','transparent'):raise ValueError('unsupported source material')
        if self.framing not in ('unknown','complete','cropped'):raise ValueError('unsupported source framing')


def source_mask_preflight(rgb, alpha, bbox, contract=None, *, overhead_scene=False,
                          glass_mode=False, require_complete=False):
    c=contract or SourceContract()
    if rgb.ndim!=3 or rgb.shape[2]!=3 or alpha.ndim!=2 or rgb.shape[:2]!=alpha.shape:
        raise ValueError('source and alpha dimensions must agree')
    if rgb.dtype!=np.uint8 or alpha.dtype!=np.uint8:raise ValueError('source and alpha must be uint8')
    x,y,w,h=bbox;H,W=alpha.shape
    if min(x,y)<0 or min(w,h)<=0 or x+w>W or y+h>H:raise ValueError('invalid matte bbox')
    warnings=[];blockers=[]
    edge=frame_edge_contact(alpha)
    n,labels,stats,_=cv2.connectedComponentsWithStats((alpha>127).astype(np.uint8),8)
    areas=stats[1:,cv2.CC_STAT_AREA].tolist()
    visible=int((alpha>127).sum());soft=int(((alpha>0)&(alpha<255)).sum())
    if visible==0:blockers.append('Empty product mask.')
    if c.pose=='unknown':warnings.append('Source camera/pose unverified; no camera compatibility claim.')
    if overhead_scene and c.pose=='unknown':
        warnings.append('Requested overhead template requires camera review; unknown source pose does not establish compatibility.')
    if overhead_scene and c.pose not in ('unknown','overhead'):
        blockers.append('Declared source pose '+c.pose+' is incompatible with an overhead scene; source-exact paste cannot change camera angle.')
    if c.pose=='suspended':warnings.append('Suspended source pose needs an explicit floating concept, not a floor-contact claim.')
    if c.material=='transparent' and not glass_mode:
        blockers.append('Transparent source needs an explicit transmission workflow; opaque RGB paste retains the old backdrop.')
    if c.material=='unknown':warnings.append('Material unverified; opaque paste may fail on transparent products.')
    if c.framing=='cropped' or edge['needs_review']:
        warnings.append('Source or alpha touches a frame cut; missing product parts cannot be recovered faithfully.')
        if require_complete:blockers.append('Complete-product requirement conflicts with a cropped source/mask.')
    if not c.genuine_tiny_text:warnings.append('No verified genuine tiny-text test; cannot claim fine-print preservation.')
    if visible and max(areas,default=0)/visible<.98:warnings.append('Multiple mask components: review every product, strap, handle and loose part; do not remove automatically.')
    if visible and soft/max(1,visible)>.3:warnings.append('Large soft-alpha fraction; inspect halos/transparency rather than certifying RGB fidelity.')
    # Cross-check bbox without changing alpha; data errors must not silently crop it.
    yy,xx=np.where(alpha>0)
    if len(xx) and (xx.min()<x or yy.min()<y or xx.max()>=x+w or yy.max()>=y+h):
        warnings.append('Nonzero alpha outside declared bbox; inspect soft-edge loss before use.')
    return dict(declaration=asdict(c),declaration_provenance='reviewer_input_not_automatic_vision',
        passed=not blockers,blockers=blockers,warnings=warnings,frame_contact=edge,
        mask_components=areas,foreground_px=visible,soft_alpha_px=soft,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
        semantic_mask_accuracy='NOT_VERIFIED',publication_status='NOT_EVALUATED')
