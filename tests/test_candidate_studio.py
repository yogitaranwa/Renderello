from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
from renderello.candidate_diagnostics import (digest, native_rgb, native_alpha,
    mask_diagnostics, compare_masks, check_regions, RegionConstraint, disagreement_image)
from renderello.candidate_harmonization import background_match, linear_over, poisson_candidate
from renderello.contact_spans import ContactPolicy, ContactSpan, render_contact, contact_map
from renderello.candidate_studio import run, load_spec, read_alpha, route_status, main


@pytest.fixture
def scene():
    rgb = np.full((96, 128, 3), 220, np.uint8)
    a = np.zeros((96, 128), np.uint8)
    a[25:60, 35:85] = 255
    rgb[a == 255] = (30, 70, 120)
    bg = np.full_like(rgb, 180)
    return rgb, a, bg


def policy(rgb, a, **kw):
    d = dict(camera='front', material='opaque', framing='complete',
        source_pose='front', source_sha256=digest(rgb), alpha_sha256=digest(a), reviewed=True)
    d.update(kw)
    return ContactPolicy(**d)


def write_inputs(tmp_path, scene):
    rgb, a, bg = scene
    for n, ar in [('source', rgb), ('mask', a), ('background', bg)]:
        Image.fromarray(ar).save(tmp_path/(n+'.png'))
    return tmp_path/'source.png', tmp_path/'mask.png', tmp_path/'background.png'


def spec(rgb, a):
    return dict(schema='renderello-candidate-studio-v1', source_pixels_sha256=digest(rgb),
        candidate_sha256={'BEN2': digest(a)}, material='opaque', framing='complete',
        source_pose='front', camera='front', mask_reviewed=True, contact_reviewed=True,
        selection='BEN2', contact_spans=[dict(name='base', x0=35, x1=85, bottom_min=59, bottom_max=59)])


def test_digest_binds_shape_dtype_and_pixels():
    a = np.arange(12, dtype=np.uint8)
    assert digest(a) != digest(a.reshape(3, 4))
    assert digest(a) != digest(a.astype(np.int16))
    assert digest(a) == digest(a.copy())
    assert digest(a.reshape(3, 4).T) == digest(a.reshape(3, 4).T.copy())


@pytest.mark.parametrize('value', [None, np.zeros((4, 4)), np.zeros((4, 4), np.uint8),
    np.zeros((4, 4, 4), np.uint8), np.zeros((2, 4, 3), np.uint8)])
def test_invalid_rgb(value):
    with pytest.raises(ValueError):
        native_rgb(value)


@pytest.mark.parametrize('value', [None, np.zeros((4, 4)), np.zeros((3, 4), np.uint8), np.zeros((4, 4, 1), np.uint8)])
def test_invalid_alpha(value):
    with pytest.raises(ValueError):
        native_alpha(value, (4, 4))


def test_rgb_allocation_guard():
    with pytest.raises(ValueError):
        native_rgb(np.zeros((2001, 2000, 3), np.uint8))


@pytest.mark.parametrize('bad', [dict(box=(-1, 0, 4, 4)), dict(box=(1, 1, 1, 2)),
    dict(box=(0, 0, 2.0, 2)), dict(kind='automatic'), dict(name=''),
    dict(threshold=0), dict(threshold=True), dict(minimum_fraction=float('nan')),
    dict(minimum_fraction=0), dict(minimum_fraction=True)])
def test_region_validation(bad):
    d = dict(name='label', box=(0, 0, 4, 4), kind='include')
    d.update(bad)
    with pytest.raises(ValueError):
        RegionConstraint(**d)


def test_regions_preserve_handle_hole_and_product(scene):
    rgb, a, _ = scene
    a[35:45, 45:55] = 0
    constraints = [RegionConstraint('body', (60, 35, 70, 45), 'include'),
                   RegionConstraint('handle-hole', (45, 35, 55, 45), 'exclude')]
    assert all(r['passed'] for r in check_regions(a, constraints))
    b = a.copy(); b[35:45, 45:55] = 255
    assert not check_regions(b, constraints)[1]['passed']
    assert np.array_equal(a[35:45, 45:55], np.zeros((10, 10), np.uint8))


def test_region_failures(scene):
    _, a, _ = scene
    c = RegionConstraint('outside', (0, 0, 200, 2), 'include')
    with pytest.raises(ValueError): check_regions(a, [c])
    c = RegionConstraint('same', (0, 0, 2, 2), 'exclude')
    with pytest.raises(ValueError): check_regions(a, [c, c])
    with pytest.raises(ValueError): check_regions(a, [c]*257)


def test_diagnostics_no_mask_mutation_and_component_holes(scene):
    rgb, a, _ = scene
    a[35:45, 45:55] = 0
    a[5:7, 5:7] = 255
    before = a.copy()
    r = mask_diagnostics(rgb, a)
    assert r['component_count'] == 2 and r['hole_count'] == 1
    assert r['hole_areas'] == [100]
    assert 'MULTICOMPONENT_REVIEW_EVERY_PART' in r['flags']
    assert r['automatic_acceptance'] is False
    assert np.array_equal(a, before)


@pytest.mark.parametrize('fill,flag', [(0, 'EMPTY_MASK'), (255, 'FULL_FRAME_FOREGROUND'),
    (40, 'NO_HIGH_ALPHA_SUPPORT')])
def test_bad_masks_flagged(scene, fill, flag):
    rgb, a, _ = scene
    a[:] = fill
    assert flag in mask_diagnostics(rgb, a)['flags']


def test_frame_crop_soft_and_color_flags(scene):
    rgb, a, _ = scene
    a[:, 0] = 255
    a[25:60, 35:85] = 180
    assert 'FRAME_TOUCH_NOT_PROOF_OF_COMPLETE_PRODUCT' in mask_diagnostics(rgb, a)['flags']
    assert 'LARGE_SOFT_BAND_NOT_OPTICAL_TRANSMISSION' in mask_diagnostics(rgb, a)['flags']
    a[25:60, 35:85] = 255
    rgb[:] = 5
    assert 'DARK_FOREGROUND_EDGE_REVIEW' in mask_diagnostics(rgb, a)['flags']
    rgb[:] = 250
    assert 'WHITE_FOREGROUND_EDGE_REVIEW' in mask_diagnostics(rgb, a)['flags']


def test_tiny_odd_aspect():
    rgb = np.zeros((4, 100, 3), np.uint8)
    a = np.zeros((4, 100), np.uint8); a[1:3, 10:20] = 255
    flags = mask_diagnostics(rgb, a)['flags']
    assert 'TINY_SOURCE_NO_DETAIL_RECOVERY' in flags and 'ODD_ASPECT_REVIEW' in flags


def test_component_json_is_bounded_not_mask():
    rgb = np.zeros((80, 80, 3), np.uint8)
    a = np.zeros((80, 80), np.uint8); a[::3, ::3] = 255
    r = mask_diagnostics(rgb, a)
    assert r['component_count'] == 729
    assert len(r['components']) == 256 and r['component_list_truncated']
    assert a.sum() == 729*255


def test_comparison_is_not_selection(scene):
    rgb, a, _ = scene
    b = a.copy(); b[2:5, 3:7] = 255
    r = compare_masks(rgb, {'prior': a, 'new': b})[0]
    assert r['right_only_pixels'] == 12
    assert r['agreement_is_accuracy'] is False and r['automatically_selected'] is None
    im = disagreement_image(a, b)
    assert (im[3, 4] == [30, 170, 255]).all()
    assert (im[30, 40] == [110, 110, 110]).all()
    assert np.count_nonzero(a) == 35*50


def test_empty_comparison_iou_unknown(scene):
    rgb, a, _ = scene
    a[:] = 0
    assert compare_masks(rgb, {'a': a, 'b': a})[0]['hard_iou'] is None
    with pytest.raises(ValueError): compare_masks(rgb, {'only': a})
    with pytest.raises(ValueError): compare_masks(rgb, {str(i): a for i in range(9)})


def test_linear_over_exact_endpoints(scene):
    rgb, a, bg = scene
    a[20, 30] = 128
    result = linear_over(rgb, a, bg)
    assert np.array_equal(result[a == 255], rgb[a == 255])
    assert np.array_equal(result[a == 0], bg[a == 0])
    with pytest.raises(ValueError): linear_over(rgb, a, bg[:, :-1])


def test_background_matching_changes_only_background_data(scene):
    rgb, a, bg = scene
    samples = np.zeros(a.shape, bool); samples[:10] = True
    original = rgb.copy()
    changed, report = background_match(bg, rgb, a, samples, reference_reviewed=True, amount=.25)
    assert (changed == 190).all()
    assert report['maximum_observed_delta'] == 10
    assert report['source_rgb_changed'] is False
    assert np.array_equal(rgb, original)
    assert np.array_equal(linear_over(rgb, a, changed)[a == 255], rgb[a == 255])


@pytest.mark.parametrize('settings', [dict(reference_reviewed=False), dict(amount=.6),
    dict(amount=float('nan')), dict(maximum_delta=True), dict(maximum_delta=0)])
def test_background_matching_limits(scene, settings):
    rgb, a, bg = scene
    samples = a == 0
    kw = dict(reference_reviewed=True); kw.update(settings)
    with pytest.raises(ValueError): background_match(bg, rgb, a, samples, **kw)


def test_background_sampling_rejects_foreground_and_small_sample(scene):
    rgb, a, bg = scene
    with pytest.raises(ValueError): background_match(bg, rgb, a, a > 0, reference_reviewed=True)
    samples = np.zeros(a.shape, bool); samples[0, :5] = True
    with pytest.raises(ValueError): background_match(bg, rgb, a, samples, reference_reviewed=True)
    with pytest.raises(ValueError): background_match(bg, rgb, a, samples.astype(np.uint8), reference_reviewed=True)


def test_poisson_real_opencv_and_protected_pixels(scene):
    rgb, a, bg = scene
    # Supply texture so this checks a real solver, not just a mocked call.
    rgb[30:40, 50:60] = (190, 100, 20)
    protected = np.zeros(a.shape, bool); protected[32:37, 52:57] = True
    out, changed, report = poisson_candidate(rgb, a, bg, protected=protected,
        color_change_permitted=True)
    baseline = linear_over(rgb, a, bg)
    assert np.array_equal(out[protected], rgb[protected])
    assert np.array_equal(out[a == 0], bg[a == 0])
    assert not changed[a == 0].any()
    assert report['changed_pixels'] == int(np.any(out != baseline, 2).sum())
    assert report['physical_relighting'] is False and report['automatic_acceptance'] is False
    assert report['changed_pixels'] > 0


def test_poisson_requires_optin_and_complete_region(scene):
    rgb, a, bg = scene
    with pytest.raises(ValueError): poisson_candidate(rgb, a, bg)
    a[0, 40] = 255
    with pytest.raises(ValueError): poisson_candidate(rgb, a, bg, color_change_permitted=True)
    a[0, 40] = 0
    protected = a == 0
    with pytest.raises(ValueError): poisson_candidate(rgb, a, bg, protected=protected, color_change_permitted=True)
    with pytest.raises(ValueError): poisson_candidate(rgb, a, bg, mode='made-up', color_change_permitted=True)


@pytest.mark.parametrize('kw', [dict(opacity=True), dict(opacity=.9), dict(decay=float('nan')),
    dict(length=0), dict(length=True), dict(cast_dx=100), dict(lateral_sigma=-1),
    dict(reviewed='yes'), dict(source_sha256='x'*64), dict(camera='side')])
def test_contact_policy_bounds(scene, kw):
    rgb, a, _ = scene
    with pytest.raises(ValueError): policy(rgb, a, **kw)


@pytest.mark.parametrize('kw', [dict(x0=-1), dict(x1=2), dict(bottom_min=60, bottom_max=59),
    dict(name=''), dict(x1=10.5)])
def test_contact_span_bounds(kw):
    d = dict(name='base', x0=35, x1=85, bottom_min=59, bottom_max=59); d.update(kw)
    with pytest.raises(ValueError): ContactSpan(**d)


def test_native_contact_attached_finite_exact(scene):
    rgb, a, bg = scene
    p = policy(rgb, a, lateral_sigma=0, cast_dx=0)
    spans = [ContactSpan('base', 35, 85, 59, 59)]
    out, shade, r = render_contact(rgb, a, bg, spans, p)
    assert shade[60, 50] == pytest.approx(p.opacity)
    assert shade[61, 50] < shade[60, 50]
    assert shade[80:, 50].sum() == 0
    assert shade[a > 0].sum() == 0
    assert np.array_equal(out[a == 255], rgb[a == 255])
    assert r['source_resampled'] is False and r['recovered_3d_geometry'] is False


def test_multiobject_contact_does_not_bridge(scene):
    rgb, a, bg = scene
    a[:] = 0; a[20:45, 15:35] = 255; a[30:60, 85:105] = 255
    p = policy(rgb, a, lateral_sigma=0, cast_dx=0)
    spans = [ContactSpan('left', 15, 35, 44, 44), ContactSpan('right', 85, 105, 59, 59)]
    _, shade, r = render_contact(rgb, a, bg, spans, p)
    assert shade[45, 20] > 0 and shade[60, 90] > 0
    assert shade[:, 40:80].sum() == 0
    assert len(r['spans']) == 2


@pytest.mark.parametrize('kw', [dict(reviewed=False), dict(material='transparent'),
    dict(material='reflective'), dict(framing='cropped'), dict(framing='unknown'),
    dict(source_pose='suspended'), dict(source_pose='unknown'), dict(camera='flatlay'),
    dict(source_sha256='0'*64), dict(alpha_sha256='1'*64)])
def test_contact_ineligible(scene, kw):
    rgb, a, _ = scene
    with pytest.raises(ValueError): contact_map(rgb, a, [ContactSpan('base', 35, 85, 59, 59)], policy(rgb, a, **kw))


def test_contact_rejects_false_coordinates_overlap_crop_and_no_room(scene):
    rgb, a, bg = scene
    p = policy(rgb, a)
    for spans in ([], [ContactSpan('empty', 0, 10, 0, 0)],
                  [ContactSpan('wrong', 35, 85, 58, 58)],
                  [ContactSpan('outside', 35, 129, 59, 59)],
                  [ContactSpan('one', 35, 70, 59, 59), ContactSpan('two', 65, 85, 59, 59)]):
        with pytest.raises(ValueError): contact_map(rgb, a, spans, p)
    with pytest.raises(ValueError): contact_map(rgb, a, [ContactSpan('base', 35, 85, 59, 59)], replace(p, length=60))
    a[0, 50] = 255
    with pytest.raises(ValueError): contact_map(rgb, a, [ContactSpan('base', 35, 85, 59, 59)], policy(rgb, a))


def test_flatlay_distance_and_no_upright_spans(scene):
    rgb, a, bg = scene
    p = policy(rgb, a, camera='flatlay', source_pose='overhead')
    _, shade, r = render_contact(rgb, a, bg, [], p)
    assert shade[60, 50] > shade[65, 50] > 0
    assert not shade[a > 0].any()
    assert r['method'] == 'flatlay_finite_outside_distance'
    with pytest.raises(ValueError): contact_map(rgb, a, [ContactSpan('base', 35, 85, 59, 59)], p)


def test_full_studio_diagnostics_and_no_selection(tmp_path, scene):
    source, mask, _ = write_inputs(tmp_path, scene)
    r = run(source, tmp_path/'out', {'BEN2': mask, 'alternative': mask})
    assert r['accepted_count'] == 0 and r['automatically_selected'] is None
    assert len(r['candidates']) == 2 and len(r['comparisons']) == 1
    assert (tmp_path/'out/DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()
    assert not (tmp_path/'out/INCOMPLETE').exists()
    with pytest.raises(FileExistsError): run(source, tmp_path/'out', {'BEN2': mask})


def test_source_bound_spec_and_full_render(tmp_path, scene):
    rgb, a, bg = scene
    source, mask, background = write_inputs(tmp_path, scene)
    s = tmp_path/'spec.json'; s.write_text(json.dumps(spec(rgb, a)))
    r = run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s, background=background)
    assert r['render_status'] == 'RESEARCH_CANDIDATE_NOT_ACCEPTED'
    assert r['contact_render']['opaque_rgb_exact']
    out = np.array(Image.open(tmp_path/'out/source-locked-contact-candidate.png'))
    assert np.array_equal(out[a == 255], rgb[a == 255])
    hashes = json.loads((tmp_path/'out/artifact-hashes.json').read_text())
    assert 'report.json' in hashes and 'source-rgb.png' in hashes


@pytest.mark.parametrize('change', [dict(material='transparent'), dict(material='reflective'),
    dict(framing='cropped'), dict(source_pose='suspended'), dict(mask_reviewed=False)])
def test_studio_blocked_not_fake_output(tmp_path, scene, change):
    rgb, a, _ = scene
    source, mask, bg = write_inputs(tmp_path, scene)
    d = spec(rgb, a); d.update(change)
    s = tmp_path/'spec.json'; s.write_text(json.dumps(d))
    r = run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s, background=bg)
    assert r['render_status'] == 'BLOCKED'
    assert r['render_blockers']
    assert not (tmp_path/'out/source-locked-contact-candidate.png').exists()


@pytest.mark.parametrize('change', [dict(schema='v2'), dict(source_pixels_sha256='0'*64),
    dict(candidate_sha256={'BEN2': '0'*64}), dict(typo=True)])
def test_spec_invalid_is_failure_not_acceptance(tmp_path, scene, change):
    rgb, a, _ = scene
    source, mask, _ = write_inputs(tmp_path, scene)
    d = spec(rgb, a); d.update(change)
    s = tmp_path/'spec.json'; s.write_text(json.dumps(d))
    with pytest.raises(ValueError): run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s)
    assert (tmp_path/'out/INCOMPLETE').exists()
    assert json.loads((tmp_path/'out/failure.json').read_text())['status'] == 'FAILED_NOT_ACCEPTED'
    assert not (tmp_path/'out/DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()


def test_bad_mask_file_shape_and_mode(tmp_path, scene):
    source, _, _ = write_inputs(tmp_path, scene)
    with pytest.raises(ValueError): read_alpha(source, scene[1].shape)
    Image.new('L', (10, 10)).save(tmp_path/'bad.png')
    with pytest.raises(ValueError): read_alpha(tmp_path/'bad.png', scene[1].shape)


def test_embedded_alpha_retained_not_revealed(tmp_path, scene):
    rgb, a, _ = scene
    source = tmp_path/'rgba.png'; Image.fromarray(np.dstack([rgb, a])).save(source)
    r = run(source, tmp_path/'out', {})
    assert list(r['candidates']) == ['embedded-alpha']
    Image.fromarray(np.full(a.shape, 255, np.uint8)).save(tmp_path/'full.png')
    with pytest.raises(ValueError): run(source, tmp_path/'bad', {'full': tmp_path/'full.png'})


def test_source_exif_normalization_then_mask_pairing(tmp_path):
    rgb = Image.new('RGB', (20, 30), 'red'); exif = Image.Exif(); exif[274] = 6
    rgb.save(tmp_path/'oriented.jpg', exif=exif)
    Image.new('L', (20, 30), 255).save(tmp_path/'wrong.png')
    with pytest.raises(ValueError): run(tmp_path/'oriented.jpg', tmp_path/'out', {'wrong': tmp_path/'wrong.png'})
    Image.new('L', (30, 20), 255).save(tmp_path/'right.png')
    r = run(tmp_path/'oriented.jpg', tmp_path/'ok', {'right': tmp_path/'right.png'})
    assert r['intake']['output_size'] == [30, 20]
    assert 'exif_orientation_6' in r['intake']['transformations']


def test_cli_real_and_duplicate_name(tmp_path, scene, capsys):
    source, mask, _ = write_inputs(tmp_path, scene)
    main(['--source', str(source), '--out', str(tmp_path/'out'), '--alpha', 'BEN2='+str(mask)])
    assert json.loads(capsys.readouterr().out)['accepted'] == 0
    with pytest.raises(SystemExit):
        main(['--source', str(source), '--out', str(tmp_path/'bad'), '--alpha', 'BEN2='+str(mask), '--alpha', 'BEN2='+str(mask)])


def test_spatial_intent_blocks_even_reviewed_candidate(tmp_path, scene):
    rgb, a, _ = scene
    source, mask, bg = write_inputs(tmp_path, scene)
    d = spec(rgb, a)
    d['regions'] = [dict(name='missing-object', box=[2, 2, 10, 10], kind='include')]
    s = tmp_path/'spec.json'; s.write_text(json.dumps(d))
    r = run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s, background=bg)
    assert 'SPATIAL_INTENT_CONSTRAINT_FAILED' in r['render_blockers']


def test_guided_matting_real_worker_integration(tmp_path, scene):
    rgb, a, _ = scene
    source, mask, _ = write_inputs(tmp_path, scene)
    from renderello.matting import make_trimap
    tri = tmp_path/'trimap.png'; Image.fromarray(make_trimap(a, 2)).save(tri)
    r = run(source, tmp_path/'out', {'BEN2': mask}, matting_candidate='BEN2', trimap=tri, matting_method='guided')
    assert set(r['candidates']) == {'BEN2', 'BEN2-refined'}
    out = np.array(Image.open(tmp_path/'out/candidates/BEN2-refined/cutout.png'))
    assert np.array_equal(out[:, :, :3], rgb)
    assert (tmp_path/'out/matting-worker/matting-manifest.json').exists()


def test_optional_model_failure_keeps_failed_status_not_fallback(tmp_path, scene, monkeypatch):
    source, mask, _ = write_inputs(tmp_path, scene)
    import renderello.bilateral_mask
    def fail(*args, **kwargs):
        raise RuntimeError('mocked worker timeout for propagation test, not inference evidence')
    monkeypatch.setattr(renderello.bilateral_mask, 'guarded_file', fail)
    with pytest.raises(RuntimeError):
        run(source, tmp_path/'out', {'BEN2': mask}, birefnet_checkpoint=tmp_path/'missing.onnx')
    assert (tmp_path/'out/INCOMPLETE').exists()
    assert not (tmp_path/'out/DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()
    assert json.loads((tmp_path/'out/failure.json').read_text())['error_type'] == 'RuntimeError'


def test_harmonization_full_pipeline_paths(tmp_path, scene):
    rgb, a, _ = scene
    source, mask, bg = write_inputs(tmp_path, scene)
    samples = np.zeros(a.shape, np.uint8); samples[:10] = 255
    Image.fromarray(samples).save(tmp_path/'samples.png')
    protected = np.zeros(a.shape, np.uint8); protected[30:40, 50:60] = 255
    Image.fromarray(protected).save(tmp_path/'protected.png')
    d = spec(rgb, a)
    d['background_match'] = dict(sample_mask=str(tmp_path/'samples.png'), reference_reviewed=True, amount=.2)
    d['poisson'] = dict(protected_mask=str(tmp_path/'protected.png'), color_change_permitted=True)
    s = tmp_path/'spec.json'; s.write_text(json.dumps(d))
    r = run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s, background=bg)
    assert r['background_harmonization']['source_rgb_changed'] is False
    assert r['poisson']['protected_rgb_exact']
    assert r['poisson']['entire_product_source_exact'] is False
    assert r['physical_relighting'] is False
    assert (tmp_path/'out/color-changing-poisson-candidate.png').exists()
    assert (tmp_path/'out/poisson-changed-pixels.png').exists()


def test_mask_alpha_exif_refused(tmp_path, scene):
    a = Image.fromarray(scene[1]); exif = Image.Exif(); exif[274] = 3
    a.save(tmp_path/'alpha.png', exif=exif)
    with pytest.raises(ValueError): read_alpha(tmp_path/'alpha.png', scene[1].shape)


def test_studio_failure_does_not_rewrite_original(tmp_path, scene):
    import hashlib
    source, mask, bg = write_inputs(tmp_path, scene)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, mask, bg)}
    with pytest.raises(ValueError): run(source, tmp_path/'out', {'../unsafe': mask})
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_source_icc_explicit_conversion_pipeline(tmp_path, scene):
    from PIL import ImageCms
    rgb, a, _ = scene
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    source = tmp_path/'profile.png'; Image.fromarray(rgb).save(source, icc_profile=profile)
    mask = tmp_path/'mask.png'; Image.fromarray(a).save(mask)
    with pytest.raises(ValueError): run(source, tmp_path/'refused', {'BEN2': mask})
    r = run(source, tmp_path/'converted', {'BEN2': mask}, allow_colour_conversion=True)
    assert 'icc_to_srgb' in r['intake']['transformations']
    assert r['quality'] == 'NOT_EVALUATED'


def test_top_level_cli_routes_candidate_command(tmp_path, scene, capsys):
    from renderello.cli import main as root_main
    source, mask, _ = write_inputs(tmp_path, scene)
    assert root_main(['candidate-studio', '--source', str(source), '--out', str(tmp_path/'out'), '--alpha', 'BEN2='+str(mask)]) == 0
    assert json.loads(capsys.readouterr().out)['candidates'] == 1


def test_soft_alpha_must_not_detach_contact_shadow(scene):
    rgb, a, bg = scene
    a[60:65, 35:85] = 60
    p = policy(rgb, a, lateral_sigma=0, cast_dx=0)
    _, shade, _ = render_contact(rgb, a, bg, [ContactSpan('base', 35, 85, 59, 59)], p)
    assert shade[60, 50] == pytest.approx(p.opacity)
    assert shade[61, 50] > 0
    assert (shade[a >= 128] == 0).all()


def test_noisy_soft_alpha_does_not_suppress_flatlay_shadow(scene):
    rgb, a, bg = scene
    a[a == 0] = 1
    p = policy(rgb, a, camera='flatlay', source_pose='overhead')
    _, shade, _ = render_contact(rgb, a, bg, [], p)
    assert shade[60, 50] > 0
    assert shade[a >= 128].sum() == 0


def test_poisson_pipeline_preserves_contact_outside_product(tmp_path, scene):
    rgb, a, bg = scene
    source, mask, background = write_inputs(tmp_path, scene)
    d = spec(rgb, a)
    d['poisson'] = dict(color_change_permitted=True)
    s = tmp_path/'spec.json'; s.write_text(json.dumps(d))
    r = run(source, tmp_path/'out', {'BEN2': mask}, spec_path=s, background=background)
    contact = np.array(Image.open(tmp_path/'out/source-locked-contact-candidate.png'))
    poisson = np.array(Image.open(tmp_path/'out/color-changing-poisson-candidate.png'))
    assert np.array_equal(poisson[a == 0], contact[a == 0])
    assert np.any(contact[a == 0] != bg[a == 0])
    assert r['render_status'] == 'RESEARCH_CANDIDATE_NOT_ACCEPTED'
