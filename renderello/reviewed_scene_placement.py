"""Actual-target-bound native placement and designed 2D contact.

Opt-in research. Support/exclusion masks and pose/light labels are caller review
hypotheses, not inferred geometry. No source relighting, resampling, glass or
photographic acceptance. A matching category is necessary, never sufficient.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .contact_spans import ContactPolicy,ContactSpan,contact_map,eligible
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear,srgb


@dataclass(frozen=True)
class ScenePlacementReview:
    source_sha256:str
    alpha_sha256:str
    scene_sha256:str
    surface_sha256:str
    exclusion_sha256:str
    source_pose:str
    target_pose:str
    source_light:str
    target_light:str
    source_note:str
    target_note:str
    source_reviewed:bool=False
    target_reviewed:bool=False

    def __post_init__(self):
        for name in ('source_sha256','alpha_sha256','scene_sha256','surface_sha256','exclusion_sha256'):
            h=getattr(self,name)
            if not isinstance(h,str)or len(h)!=64 or any(c not in '0123456789abcdef'for c in h):
                raise ValueError('exact source/alpha/scene/surface/exclusion pixel hashes required')
        for name in ('source_pose','target_pose'):
            if getattr(self,name)not in ('unknown','front','elevated','overhead','suspended'):
                raise ValueError('explicit reviewed pose category required')
        for name in ('source_light','target_light'):
            if getattr(self,name)not in ('unknown','soft_left','soft_right','overhead_soft','diffuse','hard_left','hard_right','mixed'):
                raise ValueError('explicit reviewed light category required')
        for name in ('source_note','target_note'):
            v=getattr(self,name)
            if not isinstance(v,str)or not v.strip()or len(v)>2000:raise ValueError('bounded source and actual-target review notes required')
        if type(self.source_reviewed)is not bool or type(self.target_reviewed)is not bool:
            raise ValueError('boolean source/target review declarations required')


def plan_reviewed_placement(source,alpha,scene,surface,exclusion,review,policy,spans,*,translation):
    native_rgb(source);native_alpha(alpha,source.shape[:2]);native_rgb(scene)
    h,w=alpha.shape;ch,cw=scene.shape[:2]
    if alpha.size>1_000_000 or ch*cw>2_000_000:raise ValueError('native1MPsource/2MPscene research allocation budget')
    for value in (surface,exclusion):
        if not isinstance(value,np.ndarray)or value.dtype!=np.bool_ or value.shape!=(ch,cw):
            raise ValueError('native boolean actual-scene surface/exclusion masks required')
    if not isinstance(review,ScenePlacementReview)or not isinstance(policy,ContactPolicy):
        raise ValueError('typed actual-target and contact reviews required')
    values=dict(source_sha256=source,alpha_sha256=alpha,scene_sha256=scene,surface_sha256=surface,exclusion_sha256=exclusion)
    if any(getattr(review,k)!=digest(v)for k,v in values.items()):raise ValueError('actual-target/source review binding changed')
    if policy.source_sha256!=review.source_sha256 or policy.alpha_sha256!=review.alpha_sha256:
        raise ValueError('contact policy binds different native input')
    if not isinstance(translation,(tuple,list))or len(translation)!=2 or any(type(v)is not int or v<0 for v in translation):
        raise ValueError('explicit nonnegative integer native translation required')
    if not isinstance(spans,(list,tuple))or any(not isinstance(s,ContactSpan)for s in spans):raise ValueError('typed native contact spans required')
    if len(spans)>64 or len({s.name for s in spans})!=len(spans):raise ValueError('bounded unique native contact spans required')
    if any(s.x1>w or s.bottom_max>=h for s in spans):raise ValueError('contact spans outside original source')
    dx,dy=translation;blocks=[]
    if not review.source_reviewed:blocks.append('SOURCE_REVIEW_NOT_CONFIRMED')
    if not review.target_reviewed:blocks.append('ACTUAL_TARGET_REVIEW_NOT_CONFIRMED')
    if not policy.reviewed:blocks.append('CONTACT_REVIEW_NOT_CONFIRMED')
    if policy.material!='opaque':blocks.append('OPAQUE_ONLY_NO_GLASS_OR_REFLECTION_RECOVERY')
    if policy.framing!='complete':blocks.append('COMPLETE_SOURCE_NOT_DECLARED')
    if review.source_pose in ('unknown','suspended')or review.target_pose in ('unknown','suspended'):
        blocks.append('SUPPORTED_SOURCE_TARGET_POSE_NOT_ESTABLISHED')
    elif review.source_pose!=review.target_pose:blocks.append('ACTUAL_TARGET_CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE')
    if 'unknown'in(review.source_light,review.target_light):blocks.append('SOURCE_TARGET_LIGHT_NOT_REVIEWED')
    elif review.source_light!=review.target_light:blocks.append('ACTUAL_TARGET_LIGHT_MISMATCH_NO_PRODUCT_RELIGHTING')
    expected_camera='flatlay'if review.target_pose=='overhead'else review.target_pose
    if policy.camera!=expected_camera or policy.source_pose!=review.source_pose:
        blocks.append('CONTACT_POLICY_CONTRADICTS_ACTUAL_SCENE_REVIEW')
    if not surface.any():blocks.append('NO_REVIEWED_VISIBLE_SUPPORT_SURFACE')
    if dx+w>cw or dy+h>ch:blocks.append('WHOLE_NATIVE_SOURCE_DOES_NOT_FIT_NO_RESIZE_OR_CROP')
    hard=alpha>=128
    if not hard.any()or hard.all():blocks.append('NONTRIVIAL_FOREGROUND_SUPPORT_REQUIRED')
    if any(edge.any()for edge in(hard[0],hard[-1],hard[:,0],hard[:,-1])):
        blocks.append('FRAME_TOUCHING_SUPPORT_REQUIRES_SOURCE_REVIEW')
    overlap=0;off_surface=None
    if dx+w<=cw and dy+h<=ch:
        overlap=int(((alpha>0)&exclusion[dy:dy+h,dx:dx+w]).sum())
        if overlap:blocks.append('PRODUCT_OVERLAPS_REVIEWED_SCENE_EXCLUSION')
        if review.target_pose=='overhead':
            if spans:blocks.append('FLATLAY_DOES_NOT_USE_UPRIGHT_SPANS')
            off_surface=int(((alpha>0)&~surface[dy:dy+h,dx:dx+w]).sum())
            if off_surface:blocks.append('FLATLAY_PRODUCT_FOOTPRINT_OUTSIDE_REVIEWED_SURFACE')
        elif review.target_pose in ('front','elevated'):
            if policy.opacity==0:blocks.append('UPRIGHT_VISIBLE_CONTACT_REQUIRED')
            if not spans:blocks.append('UPRIGHT_NATIVE_CONTACT_SPANS_REQUIRED')
            else:
                lower=np.where(hard.any(0),h-1-np.argmax(hard[::-1],axis=0),-1);off_surface=0
                used=set()
                for span in spans:
                    cols=np.arange(span.x0,span.x1);rows=lower[cols]
                    if any(int(c)in used for c in cols):raise ValueError('overlapping native contact spans')
                    used.update(map(int,cols))
                    if np.any((rows<span.bottom_min)|(rows>span.bottom_max)):
                        blocks.append('NATIVE_CONTACT_RANGE_CONTRADICTS_ALPHA');continue
                    cy=rows+dy;cx=cols+dx
                    if np.any(cy+1>=ch):blocks.append('CONTACT_REACHES_SCENE_FRAME');continue
                    if np.any(cy+policy.length>=ch):blocks.append('INSUFFICIENT_FINITE_SHADOW_ROOM')
                    bad=~(surface[cy,cx]&surface[cy+1,cx])|exclusion[cy+1,cx]
                    off_surface+=int(bad.sum())
                if off_surface:blocks.append('CONTACT_FOOTPRINT_OUTSIDE_REVIEWED_SURFACE')
    return dict(schema='reviewed-actual-scene-placement.v1',review=asdict(review),contact_policy=asdict(policy),
        native_spans=[asdict(s)for s in spans],translation=list(translation),source_size=[w,h],scene_size=[cw,ch],
        blockers=list(dict.fromkeys(blocks)),eligible_for_designed_preview=not blocks,
        product_exclusion_overlap_pixels=overlap,off_surface_footprint_pixels=off_surface,
        review_is_caller_declaration=True,actual_target_bytes_bound=True,recovered_geometry=False,
        source_resampled=False,source_cropped=False,product_relighting=False,accepted=False,
        limitations=['Surface, exclusions, pose and light are reviewer hypotheses, not inferred facts.',
            'Matching pose/light categories do not prove calibrated camera or physically matched illumination.',
            'Alpha completeness, edge contamination, source reflections and existing shadows remain unchanged.',
            'Designed2Dcontact and pixel attachment do not imply photographic or benchmark acceptance.'])


class PlacementRefused(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__('Actual scene placement refused: '+', '.join(report['blockers']))


def compose_reviewed_placement(source,alpha,scene,surface,exclusion,review,policy,spans,*,translation):
    report=plan_reviewed_placement(source,alpha,scene,surface,exclusion,review,policy,spans,translation=translation)
    if report['blockers']:raise PlacementRefused(report)
    # Source-native eligibility before translating, so scene padding cannot excuse
    # a source-bound contour/pose claim. No source RGB or alpha edit is made.
    eligible(source,alpha,policy)
    dx,dy=translation;h,w=alpha.shape;rgb=np.zeros_like(scene);a=np.zeros(scene.shape[:2],np.uint8)
    rgb[dy:dy+h,dx:dx+w]=source;a[dy:dy+h,dx:dx+w]=alpha
    cp=replace(policy,source_sha256=digest(rgb),alpha_sha256=digest(a))
    shifted=[ContactSpan(s.name,s.x0+dx,s.x1+dx,s.bottom_min+dy,s.bottom_max+dy)for s in spans]
    shade,contact=contact_map(rgb,a,shifted,cp);outside=(~surface)|exclusion;clipped=shade.copy();clipped[outside]=0
    if policy.camera!='flatlay':
        hard=a>=128;lower=np.where(hard.any(0),a.shape[0]-1-np.argmax(hard[::-1],axis=0),-1)
        for span in shifted:
            cols=np.arange(span.x0,span.x1)
            if not np.all(clipped[lower[cols]+1,cols]>0):raise ValueError('designed contact detached after target clipping')
    out=linear_over(rgb,a,srgb(linear(scene)*(1-clipped[...,None])))
    if not np.array_equal(out[a==255],rgb[a==255])or np.any(clipped[a>=128]):raise RuntimeError('native source/shadow invariant failed')
    report.update(contact=contact,shadow_clipped_pixels=int(((shade>0)&outside).sum()),shadow_outside_reviewed_surface_pixels=0,
        opaque_rgb_exact_pixels=int((a==255).sum()),result_sha256=digest(out),shadow_sha256=digest(clipped),
        alpha_changed=False,product_rgb_repainted=False,photographic_acceptance='NOT_EVALUATED')
    return out,clipped,report
