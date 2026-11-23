"""Executable multi-candidate studio: normalize, inspect, refine and render.

Run `python -m renderello.candidate_studio --help`. Defaults produce diagnostics,
not a selected mask or publishable image. Existing estimated masks are separately
retained. Review declarations are bound to exact normalized source/mask pixels;
they are local engineering inputs, never authenticated user approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .candidate_diagnostics import (digest, native_rgb, native_alpha,
    mask_diagnostics, compare_masks, check_regions, RegionConstraint,
    disagreement_image)
from .image_intake import inspect_image, IntakePolicy, save_intake
from .candidate_harmonization import linear_over, background_match, poisson_candidate
from .contact_spans import ContactSpan, ContactPolicy, render_contact


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def _name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('candidate name must be a safe short identifier')
    return name


def read_alpha(path, shape):
    # An alpha is not a photograph: conversion of RGB/16-bit/P inputs could hide
    # swapped files or quantization. Orientation must already match normalized RGB.
    with Image.open(path) as im:
        if im.mode != 'L' or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('single-frame L alpha required; no mode guessing')
        if im.size != (shape[1], shape[0]):
            raise ValueError('mask size differs from normalized source')
        if im.getexif().get(274, 1) != 1:
            raise ValueError('alpha has EXIF transform; normalize coordinates explicitly')
        a = np.array(im)
    return native_alpha(a, shape)


def load_spec(path, rgb, candidates):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'schema', 'source_pixels_sha256', 'candidate_sha256', 'regions',
               'selection', 'material', 'framing', 'source_pose', 'camera',
               'mask_reviewed', 'contact_reviewed', 'contact_spans', 'contact_options',
               'background_match', 'poisson'}
    if not isinstance(data, dict) or set(data)-allowed:
        raise ValueError('unknown studio specification fields')
    if data.get('schema') != 'renderello-candidate-studio-v1':
        raise ValueError('unsupported studio specification')
    if data.get('source_pixels_sha256') != digest(rgb):
        raise ValueError('specification belongs to different normalized source pixels')
    if set(data.get('candidate_sha256', {})) != set(candidates):
        raise ValueError('specification must bind exactly the supplied candidates')
    for name, a in candidates.items():
        if data['candidate_sha256'][name] != digest(a):
            raise ValueError('specification alpha changed: '+name)
    regions = []
    for item in data.get('regions', []):
        c = dict(item)
        c['box'] = tuple(c['box'])
        regions.append(RegionConstraint(**c))
    return data, regions


def write_candidate(root, name, rgb, alpha):
    directory = root / _name(name)
    directory.mkdir()
    Image.fromarray(alpha).save(directory/'alpha.png')
    Image.fromarray(np.dstack([rgb, alpha])).save(directory/'cutout.png')
    for label, color in [('dark', (20, 20, 20)), ('white', (245, 245, 245)), ('magenta', (190, 0, 140))]:
        bg = np.empty_like(rgb)
        bg[:] = color
        Image.fromarray(linear_over(rgb, alpha, bg)).save(directory/(label+'.png'))
    return directory


def route_status(spec, diagnostics, region_results):
    """Explain blockers without deleting failed candidates or fabricating fields."""
    blocked = []
    if spec is None:
        return ['SOURCE_BOUND_REVIEW_SPEC_REQUIRED']
    if spec.get('mask_reviewed') is not True:
        blocked.append('VISUAL_MASK_REVIEW_REQUIRED')
    if spec.get('material') not in ('opaque', 'transparent', 'reflective', 'unknown'):
        blocked.append('MATERIAL_DECLARATION_REQUIRED')
    elif spec['material'] == 'transparent':
        blocked.append('TRANSMISSION_FLOW_CLEAN_PLATE_REQUIRED_NOT_ALPHA')
    elif spec['material'] == 'reflective':
        blocked.append('REFLECTION_MATCH_UNVERIFIED_SOURCE_LOCK_NOT_RELIGHTING')
    elif spec['material'] == 'unknown':
        blocked.append('MATERIAL_UNKNOWN')
    if spec.get('framing') != 'complete':
        blocked.append('COMPLETE_SOURCE_NOT_REVIEWED_NO_INVENTED_PARTS')
    if spec.get('source_pose') in (None, 'unknown', 'suspended'):
        blocked.append('POSE_NOT_ELIGIBLE_FOR_SURFACE_CONTACT')
    if any(not r['passed'] for r in region_results):
        blocked.append('SPATIAL_INTENT_CONSTRAINT_FAILED')
    severe = {'EMPTY_MASK', 'FULL_FRAME_FOREGROUND', 'NO_HIGH_ALPHA_SUPPORT',
              'FRAME_TOUCH_NOT_PROOF_OF_COMPLETE_PRODUCT'}
    blocked.extend(flag for flag in diagnostics['flags'] if flag in severe)
    return blocked


def run(source, output, masks, *, spec_path=None, background=None,
        allow_colour_conversion=False, birefnet_checkpoint=None,
        matting_candidate=None, trimap=None, matting_method='cf', native_results=None, rgba_results=None,
        instance_reviews=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve prior attempt; choose a fresh output directory')
    intake = inspect_image(source, IntakePolicy(max_pixels=4_000_000,
        allow_colour_conversion=allow_colour_conversion))
    try:
        rgb = np.array(intake.image.convert('RGB'))
        native_rgb(rgb)
        source_alpha = np.array(intake.image.getchannel('A')) if intake.image.mode == 'RGBA' else None
        output.mkdir(parents=True)
        (output/'INCOMPLETE').touch()
        save_intake(intake, output/'input')
        # Always use orientation/color-normalized RGB for all downstream workers.
        normalized_source = output/'source-rgb.png'
        Image.fromarray(rgb).save(normalized_source)
        candidates = {}
        for name, path in masks.items():
            _name(name)
            candidates[name] = read_alpha(path, rgb.shape[:2])
        if not candidates and source_alpha is not None:
            candidates['embedded-alpha'] = source_alpha.copy()
        if birefnet_checkpoint is not None:
            if source_alpha is not None and np.any(source_alpha != 255):
                raise ValueError('model RGB interpretation of transparent source requires an explicit flattening policy')
            if 'BiRefNet' in candidates:
                raise ValueError('duplicate BiRefNet candidate')
            from .bilateral_mask import guarded_file
            guarded_file(normalized_source, output/'birefnet-worker', Path(birefnet_checkpoint))
            candidates['BiRefNet'] = read_alpha(output/'birefnet-worker/estimated-alpha.png', rgb.shape[:2])
        if not 1 <= len(candidates) <= 8:
            raise ValueError('supply 1..8 candidates or an embedded source alpha')
        if source_alpha is not None:
            for a in candidates.values():
                if np.any(a > source_alpha):
                    raise ValueError('candidate exceeds embedded source alpha; no hidden RGB reveal or silent multiplication')
        refinement_reviews = {}
        if matting_candidate is not None:
            if matting_candidate not in candidates or trimap is None:
                raise ValueError('matting needs a named initial candidate and explicit reviewed trimap')
            name = _name(matting_candidate+'-refined')
            if name in candidates:
                raise ValueError('refined candidate name collides')
            if len(candidates) >= 8:
                raise ValueError('no room for additional refined candidate')
            from .matte_change_review import audit_trimap, review_change
            trimap_array = read_alpha(trimap, rgb.shape[:2])
            seed_review, seed_conflicts = audit_trimap(candidates[matting_candidate], trimap_array)
            _json(output/'trimap-review.json', seed_review)
            Image.fromarray(seed_conflicts.astype(np.uint8)*255).save(output/'trimap-seed-conflicts.png')
            initial = output/'initial-for-matting.png'
            Image.fromarray(candidates[matting_candidate]).save(initial)
            from .matting import guarded_file
            guarded_file(normalized_source, initial, output/'matting-worker',
                trimap_path=Path(trimap), method=matting_method, edge_colors=False)
            candidates[name] = read_alpha(output/'matting-worker/refined-alpha.png', rgb.shape[:2])
            delta_review, delta_image = review_change(candidates[matting_candidate], candidates[name])
            refinement_reviews[name] = dict(seeds=seed_review, changes=delta_review)
            _json(output/'matte-change-review.json', delta_review)
            Image.fromarray(delta_image).save(output/'matte-alpha-change.png')
            if source_alpha is not None and np.any(candidates[name] > source_alpha):
                raise ValueError('refined candidate exceeds embedded source alpha')
        imported_results = {}
        for name, request in (native_results or {}).items():
            _name(name)
            if name in candidates or len(candidates) >= 8:
                raise ValueError('imported candidate collides or exceeds eight-candidate budget')
            if not isinstance(request, dict) or set(request) != {'directory', 'initial'}:
                raise ValueError('native import needs directory and initial candidate')
            initial = request['initial']
            if initial not in candidates:
                raise ValueError('native import initial alpha candidate is missing')
            from .native_result_import import load_verified_result
            alpha, evidence = load_verified_result(request['directory'], rgb, candidates[initial])
            if source_alpha is not None and np.any(alpha > source_alpha):
                raise ValueError('imported candidate exceeds embedded source alpha')
            candidates[name] = alpha
            refinement_reviews[name] = dict(seeds=evidence['seeds'], changes=evidence['changes'])
            imported_results[name] = evidence
            _json(output/('native-import-'+name+'.json'), evidence)
        rgba_imports = {}
        for name, request in (rgba_results or {}).items():
            _name(name)
            if name in candidates or len(candidates) >= 8:
                raise ValueError('imported RGBA candidate collides or exceeds eight-candidate budget')
            if not isinstance(request, dict) or set(request) != {'path', 'sha256', 'initial'}:
                raise ValueError('RGBA import needs path, sha256 and initial candidate')
            if request['initial'] not in candidates:
                raise ValueError('RGBA import initial alpha candidate is missing')
            from .source_locked_rgba_import import load_source_locked_rgba
            from .matte_change_review import review_change
            alpha, evidence = load_source_locked_rgba(request['path'], rgb, request['sha256'])
            if source_alpha is not None and np.any(alpha > source_alpha):
                raise ValueError('imported RGBA exceeds embedded source alpha')
            change, _ = review_change(candidates[request['initial']], alpha)
            candidates[name] = alpha
            refinement_reviews[name] = dict(changes=change)
            rgba_imports[name] = evidence
            _json(output/('rgba-import-'+name+'.json'), evidence)
        spec, regions = load_spec(spec_path, rgb, candidates) if spec_path else (None, [])
        # Optional API declarations bind exactly every final candidate, not only
        # the selected one. A failed core remains available for diagnostics but
        # cannot enter background rendering. No automatic instance detection.
        if instance_reviews is not None and (not isinstance(instance_reviews, dict)
                or set(instance_reviews) != set(candidates)):
            raise ValueError('instance reviews must bind exactly every final candidate')
        root = output/'candidates'
        root.mkdir()
        report = dict(schema='renderello-candidate-studio-result-v1',
            source_encoded_sha256=intake.record['source_sha256'],
            normalized_source_sha256=digest(rgb), intake=intake.record,
            candidates={}, comparisons=compare_masks(rgb, candidates) if len(candidates) > 1 else [],
            automatically_selected=None, accepted_count=0, quality='NOT_EVALUATED',
            physical_relighting=False, real_glass_validated=False,
            refinement_reviews=refinement_reviews, native_result_imports=imported_results,
            source_locked_rgba_imports=rgba_imports)
        for name, a in candidates.items():
            directory = write_candidate(root, name, rgb, a)
            diagnostic = mask_diagnostics(rgb, a)
            region_results = check_regions(a, regions)
            blocked = route_status(spec, diagnostic, region_results)
            if name in refinement_reviews:
                for stage, review in refinement_reviews[name].items():
                    blocked.extend('REFINEMENT_'+stage.upper()+'_'+flag for flag in review['blockers'])
            presence = None
            if instance_reviews is not None:
                from .reviewed_instance_presence import review_instance_presence
                request = instance_reviews[name]
                if not isinstance(request, dict) or set(request) != {'regions', 'bindings', 'reviewed', 'review_note'}:
                    raise ValueError('exact reviewed instance-presence request required')
                presence = review_instance_presence(rgb, np.dstack([rgb, a]), **request)
                blocked.extend('INSTANCE_PRESENCE_'+flag for flag in presence['blockers'])
                _json(directory/'instance-presence.json', presence)
            record = dict(diagnostics=diagnostic, constraints=region_results, render_blockers=blocked)
            if presence is not None:
                record['instance_presence'] = presence
            report['candidates'][name] = record
            _json(directory/'diagnostics.json', record)
        for row in report['comparisons']:
            Image.fromarray(disagreement_image(candidates[row['left']], candidates[row['right']])).save(
                output/('disagreement-'+row['left']+'-'+row['right']+'.png'))
        selected = spec.get('selection') if spec else None
        if background is not None:
            if selected not in candidates:
                raise ValueError('render requires a source-bound explicit candidate selection')
            blocked = report['candidates'][selected]['render_blockers']
            if blocked:
                report['render_status'] = 'BLOCKED'
                report['render_blockers'] = blocked
            else:
                # Strict source-sized scene. No automatic crop/resize of either.
                with Image.open(background) as im:
                    if im.mode != 'RGB' or im.size != (rgb.shape[1], rgb.shape[0]) or im.info.get('icc_profile') or im.getexif().get(274, 1) != 1:
                        raise ValueError('normalized native-sized unprofiled RGB scene required')
                    bg = np.array(im)
                a = candidates[selected]
                policy = ContactPolicy(camera=spec['camera'], material=spec['material'],
                    framing=spec['framing'], source_pose=spec['source_pose'],
                    source_sha256=digest(rgb), alpha_sha256=digest(a),
                    reviewed=spec.get('contact_reviewed', False), **spec.get('contact_options', {}))
                spans = [ContactSpan(**s) for s in spec.get('contact_spans', [])]
                if spec.get('background_match'):
                    settings = dict(spec['background_match'])
                    sample_path = settings.pop('sample_mask')
                    samples = read_alpha(sample_path, a.shape)
                    if not np.isin(samples, [0, 255]).all():
                        raise ValueError('background sample mask must be binary')
                    bg, harmonization = background_match(bg, rgb, a, samples == 255, **settings)
                    Image.fromarray(bg).save(output/'harmonized-background.png')
                    report['background_harmonization'] = harmonization
                result, shade, meta = render_contact(rgb, a, bg, spans, policy)
                Image.fromarray(result).save(output/'source-locked-contact-candidate.png')
                Image.fromarray(np.rint(shade*255).astype(np.uint8)).save(output/'contact-attenuation.png')
                report['contact_render'] = meta
                report['render_status'] = 'RESEARCH_CANDIDATE_NOT_ACCEPTED'
                if spec.get('poisson'):
                    settings = dict(spec['poisson'])
                    protected_path = settings.pop('protected_mask', None)
                    protected = read_alpha(protected_path, a.shape) if protected_path else None
                    if protected is not None and not np.isin(protected, [0, 255]).all():
                        raise ValueError('protected mask must be binary')
                    from .checked_research.contact_camera import linear, srgb
                    shaded_bg = srgb(linear(bg)*(1-shade[..., None]))
                    result, changed, meta = poisson_candidate(rgb, a, shaded_bg,
                        protected=protected == 255 if protected is not None else None, **settings)
                    Image.fromarray(result).save(output/'color-changing-poisson-candidate.png')
                    Image.fromarray(changed.astype(np.uint8)*255).save(output/'poisson-changed-pixels.png')
                    report['poisson'] = meta
        _json(output/'report.json', report)
        files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in output.rglob('*') if p.is_file() and p.name != 'INCOMPLETE'}
        _json(output/'artifact-hashes.json', files)
        (output/'INCOMPLETE').unlink()
        (output/'DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').touch()
        return report
    except Exception as exc:
        if output.is_dir():
            _json(output/'failure.json', dict(status='FAILED_NOT_ACCEPTED',
                error_type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        intake.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--alpha', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--spec', type=Path)
    p.add_argument('--background', type=Path)
    p.add_argument('--birefnet-checkpoint', type=Path)
    p.add_argument('--allow-colour-conversion', action='store_true')
    p.add_argument('--native-result', action='append', default=[], metavar='NAME=INITIAL=PATH',
        help='Replay and import a saved native-matting result, never automatically accept it')
    p.add_argument('--rgba-result', action='append', default=[], metavar='NAME=INITIAL=SHA256=PATH',
        help='Verify pinned native RGBA against source RGB; never automatically accept it')
    p.add_argument('--matting-candidate')
    p.add_argument('--trimap', type=Path)
    p.add_argument('--matting-method', choices=['cf', 'lkm', 'guided'], default='cf')
    a = p.parse_args(argv)
    masks = {}
    for item in a.alpha:
        if '=' not in item:
            p.error('--alpha needs NAME=PATH')
        name, path = item.split('=', 1)
        if name in masks:
            p.error('duplicate candidate name')
        masks[name] = Path(path)
    native_results = {}
    for item in a.native_result:
        fields = item.split('=', 2)
        if len(fields) != 3 or not all(fields):
            p.error('--native-result needs NAME=INITIAL=PATH')
        name, initial, path = fields
        if name in native_results:
            p.error('duplicate native-result candidate name')
        native_results[name] = dict(initial=initial, directory=Path(path))
    rgba_results = {}
    for item in a.rgba_result:
        fields = item.split('=', 3)
        if len(fields) != 4 or not all(fields):
            p.error('--rgba-result needs NAME=INITIAL=SHA256=PATH')
        name, initial, sha256, path = fields
        if name in rgba_results:
            p.error('duplicate RGBA candidate name')
        rgba_results[name] = dict(initial=initial, sha256=sha256, path=Path(path))
    report = run(a.source, a.out, masks, spec_path=a.spec, background=a.background,
        allow_colour_conversion=a.allow_colour_conversion,
        birefnet_checkpoint=a.birefnet_checkpoint, matting_candidate=a.matting_candidate,
        trimap=a.trimap, matting_method=a.matting_method, native_results=native_results,
        rgba_results=rgba_results)
    print(json.dumps(dict(status='diagnostics_complete_not_accepted',
        candidates=len(report['candidates']), accepted=0)))


if __name__ == '__main__':
    main()
