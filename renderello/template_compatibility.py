"""Source-bound draft-template feasibility, not scene or photographic acceptance.

Profiles are explicit readings of this draft library's requests, not vision.
No camera, source lighting, pose, geometry or transparent optics is invented.
"""
from dataclasses import dataclass, asdict
from types import MappingProxyType
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .schemas import Aspect
from .templates import TEMPLATES


@dataclass(frozen=True)
class TemplateNeeds:
    camera: str
    support: str
    light: str
    reflection: bool = False
    rotated_product: bool = False


NEEDS = MappingProxyType({
 'clean_studio': TemplateNeeds('front','floor','front_left_soft'),
 'luxury_hero': TemplateNeeds('low','floor','top_rim',True),
 # Three-quarter is an azimuth convention here, not evidence of elevation.
 # The text also says product height, so do not silently choose an elevated
 # camera. Require disambiguation before source-locked preview.
 'editorial_beauty': TemplateNeeds('unknown','platform','left_soft'),
 'lifestyle_kitchen': TemplateNeeds('front','floor','rear_left_window'),
 'natural_outdoor': TemplateNeeds('low','floor','open_shade'),
 'minimal_tabletop': TemplateNeeds('front','platform','overhead_soft'),
 'seasonal_holiday': TemplateNeeds('elevated','floor','warm_mixed'),
 'social_ad': TemplateNeeds('suspended','none','graphic_hard',False,True),
 'reflective_surface': TemplateNeeds('low','floor','symmetric_side',True),
 'flat_lay': TemplateNeeds('overhead','flatlay','overhead_soft'),
})
assert set(NEEDS) == set(TEMPLATES)


@dataclass(frozen=True)
class SourceSceneReview:
    source_sha256: str
    alpha_sha256: str
    pose: str
    material: str
    framing: str
    support: str
    light: str
    note: str
    reviewed: bool = False

    def __post_init__(self):
        for h in (self.source_sha256,self.alpha_sha256):
            if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef'for c in h):
                raise ValueError('native source/alpha pixel SHA256 required')
        choices=dict(pose=('unknown','front','low','elevated','overhead','suspended'),
            material=('unknown','opaque','transparent','reflective'),
            framing=('unknown','complete','cropped'),
            support=('unknown','floor','platform','flatlay','none'),
            light=('unknown','front_left_soft','top_rim','left_soft','rear_left_window',
                   'open_shade','overhead_soft','warm_mixed','graphic_hard','symmetric_side'))
        for n,allowed in choices.items():
            if getattr(self,n) not in allowed:raise ValueError('unsupported source '+n)
        if type(self.reviewed)is not bool:raise ValueError('boolean review declaration required')
        if not isinstance(self.note,str)or not self.note.strip()or len(self.note)>2000:
            raise ValueError('bounded source/lighting review note required')


def plan_template(rgb,alpha,template,review,*,aspect=None):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('native1MP review budget; no downscale')
    if template not in NEEDS:raise ValueError('registered draft template required')
    if not isinstance(review,SourceSceneReview):raise ValueError('typed source/scene review required')
    if review.source_sha256!=digest(rgb)or review.alpha_sha256!=digest(alpha):
        raise ValueError('review belongs to different source/alpha')
    selected=TEMPLATES[template].aspect if aspect is None else Aspect(aspect)
    need=NEEDS[template];blockers=[];warnings=[]
    if not review.reviewed:blockers.append('SOURCE_REVIEW_NOT_CONFIRMED')
    if review.material!='opaque':blockers.append('OPAQUE_ONLY_NO_GLASS_OR_REFLECTIVE_RECOVERY')
    if review.framing!='complete':blockers.append('COMPLETE_SOURCE_NOT_ESTABLISHED')
    if need.camera=='unknown':blockers.append('TEMPLATE_CAMERA_ELEVATION_NOT_DISAMBIGUATED')
    elif review.pose=='unknown':blockers.append('SOURCE_CAMERA_ELEVATION_UNREVIEWED')
    elif review.pose!=need.camera:blockers.append('CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE')
    # Floor/platform contact types differ: a prior floor cannot certify support
    # on a newly authored plinth. Its new footprint needs its own surface review.
    if review.support!=need.support:blockers.append('SUPPORT_MISMATCH_REQUIRES_NEW_SURFACE_REVIEW')
    if review.light=='unknown':blockers.append('SOURCE_LIGHT_UNREVIEWED')
    elif review.light!=need.light:blockers.append('LIGHT_MISMATCH_NO_PRODUCT_RELIGHTING')
    if need.reflection:blockers.append('REQUESTED_REFLECTION_NOT_RECOVERED')
    if need.rotated_product:blockers.append('REQUESTED_ROTATION_NOT_SOURCE_LOCKED')
    support=alpha>0;hard=alpha>=128
    if not hard.any():blockers.append('NO_HARD_FOREGROUND')
    if any(v.any()for v in(support[0],support[-1],support[:,0],support[:,-1])):
        blockers.append('FRAME_TOUCHING_ALPHA_REQUIRES_REVIEW')
    w,h=selected.size;sh,sw=alpha.shape
    # Entire native source is retained, including soft support. No hidden crop.
    translation=((w-sw)//2,(h-sh)//2)if sw<=w and sh<=h else None
    if translation is None:blockers.append('NATIVE_SOURCE_DOES_NOT_FIT_ASPECT_NO_RESIZE')
    if need.support!='none':warnings.append('Actual target surface and every product contact footprint still need raster/visual review.')
    warnings.extend(['Matching reviewed categories does not prove matched physical lighting or perspective.',
        'Template is a draft, not an evaluated polished cover or publish-usable output.',
        'No source alpha correction, edge RGB recovery, optical glass, product relighting or reflection is performed.'])
    return dict(schema='source-template-plan.v1',template=template,template_name=TEMPLATES[template].name,
        template_state='DRAFT_NOT_PHOTOGRAPHICALLY_ACCEPTED',needs=asdict(need),source_review=asdict(review),
        camera_requirement_text=TEMPLATES[template].camera,
        camera_semantics='COARSE_ELEVATION_ONLY_NOT_AZIMUTH_FOCAL_LENGTH_OR_PERSPECTIVE_MATCH',
        review_is_caller_declaration=True,source_size=[sw,sh],aspect=selected.value,canvas=[w,h],
        whole_source_translation=list(translation)if translation is not None else None,
        blockers=blockers,warnings=warnings,eligible_for_reviewed_preview=not blockers,
        source_rgb_locked=True,product_light_changed=False,source_resampled=False,
        glass_recovered=False,reflection_recovered=False,photo_accepted=False,
        actual_target_scene_verified=False,template_polish_verified=False,default_changed=False)
