"""Verified saved native matting to reviewed 2D contact comparison.
No new inference, pixel scaling, pose change, scene-light inference or acceptance.
"""
from dataclasses import asdict, replace
import numpy as np
from .candidate_diagnostics import digest, native_rgb
from .native_result_import import load_verified_result
from .contact_spans import ContactPolicy, ContactSpan, render_contact
from .candidate_harmonization import linear_over


def compare_saved_result_contact(directory, source, initial_alpha, background, spans, policy,
        *, support_floor_row=None, support_tolerance=2, translation=(0,0)):
    if not isinstance(policy, ContactPolicy):
        raise ValueError('typed source-bound contact policy required')
    expected_source_pose = 'overhead' if policy.camera == 'flatlay' else policy.camera
    if expected_source_pose != policy.source_pose:
        raise ValueError('this comparison requires compatible declared source and target camera')
    flatlay = policy.camera == 'flatlay'
    if flatlay and (support_floor_row is not None or spans):
        raise ValueError('flatlay uses the full silhouette; no upright spans or floor row')
    alpha, replay = load_verified_result(directory, source, initial_alpha)
    if digest(source) != policy.source_sha256 or digest(alpha) != policy.alpha_sha256:
        raise ValueError('scene declarations do not bind to this verified native result')
    native_rgb(background)
    if not isinstance(translation,(tuple,list)) or len(translation)!=2 or any(type(v) is not int or v<0 for v in translation):
        raise ValueError('nonnegative integer source-to-canvas translation required')
    dx,dy=translation
    h,w=source.shape[:2];ch,cw=background.shape[:2]
    if dx+w>cw or dy+h>ch:
        raise ValueError('entire native source must fit canvas; no crop or resize')
    original_policy=asdict(policy)
    if not flatlay and (type(support_floor_row) is not int or not 0 <= support_floor_row < alpha.shape[0]-1):
        raise ValueError('explicit native support floor row required')
    if source.shape != background.shape or dx or dy:
        translated=np.zeros(background.shape,np.uint8);translated_alpha=np.zeros((ch,cw),np.uint8)
        translated[dy:dy+h,dx:dx+w]=source;translated_alpha[dy:dy+h,dx:dx+w]=alpha
        source,alpha=translated,translated_alpha
        if any(not isinstance(s,ContactSpan) for s in spans):raise ValueError('typed native source spans required')
        spans=[ContactSpan(s.name,s.x0+dx,s.x1+dx,s.bottom_min+dy,s.bottom_max+dy) for s in spans]
        policy=replace(policy,source_sha256=digest(source),alpha_sha256=digest(alpha))
    native_support_floor_row=support_floor_row
    if not flatlay:
        support_floor_row+=dy
    final, shade, contact = render_contact(source, alpha, background, spans, policy)
    no_contact = linear_over(source, alpha, background)
    hard = alpha >= 128
    lower = np.where(hard.any(0), alpha.shape[0]-1-np.argmax(hard[::-1],0), -1)
    if not flatlay and (type(support_floor_row) is not int or not 0 <= support_floor_row < alpha.shape[0]-1):
        raise ValueError('explicit native support floor row required')
    if type(support_tolerance) is not int or not 0 <= support_tolerance <= 4:
        raise ValueError('support tolerance must be integer 0..4 pixels')
    attachments = []
    for span in spans:
        cols = np.arange(span.x0,span.x1)
        if np.any(np.abs(lower[cols]-support_floor_row) > support_tolerance):
            raise ValueError('reviewed support contradicts declared floor row; no shadow under elevated contour')
        attached = shade[lower[cols]+1,cols] > 0
        attachments.append(dict(name=span.name, columns=int(len(cols)),
            first_row_shaded_columns=int(attached.sum()), all_attached=bool(attached.all())))
    if attachments and not all(a['all_attached'] for a in attachments):
        raise ValueError('designed contact has a detached first row')
    if np.any(shade[hard]) or not np.array_equal(final[alpha==255],source[alpha==255]):
        raise RuntimeError('hard-support shadow exclusion/source opaque invariant failed')
    record=dict(schema='saved-native-contact-comparison.v1', import_replay=replay,
        contact=contact, policy=asdict(policy), attachments=attachments,
        support_floor_row=support_floor_row, support_tolerance=support_tolerance,
        source_policy=original_policy, native_support_floor_row=native_support_floor_row,
        source_to_canvas_translation=[dx,dy], original_source_size=[w,h],
        source_cropped=False,
        support_plane='REVIEWED_FLATLAY_SILHOUETTE' if flatlay else 'EXPLICIT_REVIEW_NOT_RECOVERED_GEOMETRY',
        source_target_camera_compatible=True,
        no_contact_sha256=digest(no_contact), result_sha256=digest(final),
        source_rgb_repainted=False, alpha_repainted=False, source_resampled=False,
        opaque_source_pixels_exact=True, shadow_on_hard_support_pixels=int(np.count_nonzero(shade[hard])),
        assistant_visual_review_required=True, accepted=False, independent_review=False,
        limitations=['Material, completeness, camera and contact spans are reviewed declarations, not inferred facts.',
          'Same source lighting/reflections and alpha errors survive.',
          'Pixel attachment does not establish plausible 3D support, scene light or publication quality.',
          'Saved model execution provenance remains a manifest claim; numeric replay proves consistency only.'])
    return no_contact, final, shade, record
