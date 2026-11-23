import copy
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_instance_presence import review_instance_presence, require_instance_presence

def fixture():
    rgb = np.full((20, 30, 3), 110, np.uint8)
    a = np.full((20, 30), 255, np.uint8)
    rgba = np.dstack([rgb, a])
    m = np.zeros(a.shape, bool)
    m[3:12, 4:14] = True
    r = dict(name='white_bottle', mask=m, mask_sha256=digest(m), intent='retain', material='opaque', alpha_threshold=200, covered_fraction=0.95, mean_alpha=0.9)
    return (rgb, rgba, [r])

def call(rgb, rgba, regions, fn=review_instance_presence, **kw):
    args = dict(bindings=dict(source=digest(rgb), candidate=digest(rgba)), reviewed=True, review_note='Explicit source interior and intent, not object detection.')
    args.update(kw)
    return fn(rgb, rgba, regions, **args)

def test_pass_is_not_acceptance_and_no_mutation():
    (rgb, rgba, r) = fixture()
    before = [x.copy() for x in [rgb, rgba, r[0]['mask']]]
    out = call(rgb, rgba, r, require_instance_presence)
    assert out['declared_coverage_satisfied'] and (not out['photo_accepted']) and (not out['instance_detection']) and (not out['transparent_optics_verified'])
    assert all((np.array_equal(x, y) for (x, y) in zip(before, [rgb, rgba, r[0]['mask']])))

def test_rgb_identity_alone_cannot_hide_missing_instance():
    (rgb, rgba, r) = fixture()
    rgba[r[0]['mask'], 3] = 0
    out = call(rgb, rgba, r)
    assert out['source_rgb_changed_pixels'] == 0 and (not out['declared_coverage_satisfied'])
    with pytest.raises(ValueError) as e:
        call(rgb, rgba, r, require_instance_presence)
    assert len(e.value.review_report['blockers']) == 2

def test_mean_detects_ghost_even_with_coverage_above_threshold():
    (rgb, rgba, r) = fixture()
    r[0]['alpha_threshold'] = 1
    rgba[r[0]['mask'], 3] = 2
    out = call(rgb, rgba, r)
    assert out['regions'][0]['observed_covered_fraction'] == 1 and out['blockers'] == ['white_bottle:RETAIN_MEAN_ALPHA_LOW']

def test_aggregate_cannot_hide_one_lost_instance():
    (rgb, rgba, r) = fixture()
    m = np.zeros((20, 30), bool)
    m[13:18, 20:25] = True
    q = dict(r[0], name='brush', mask=m, mask_sha256=digest(m))
    r.append(q)
    rgba[m, 3] = 0
    out = call(rgb, rgba, r)
    assert out['regions'][0]['constraint_satisfied'] and (not out['regions'][1]['constraint_satisfied'])

def test_explicit_exclusion_and_transparent_scope():
    (rgb, rgba, r) = fixture()
    r[0].update(intent='exclude', material='transparent', covered_fraction=0, mean_alpha=0)
    assert not call(rgb, rgba, r)['declared_coverage_satisfied']
    rgba[r[0]['mask'], 3] = 0
    out = call(rgb, rgba, r)
    assert out['declared_coverage_satisfied'] and (not out['physical_opacity_verified'])

@pytest.mark.parametrize('where', [(0, 0), (5, 5)])
def test_repaint_even_unreviewed_hidden_is_blocked(where):
    (rgb, rgba, r) = fixture()
    rgba[where[0], where[1], 0] += 1
    assert 'CANDIDATE_SOURCE_RGB_CHANGED' in call(rgb, rgba, r)['blockers']

@pytest.mark.parametrize('field,value', [('alpha_threshold', True), ('alpha_threshold', 0), ('alpha_threshold', 256), ('covered_fraction', float('nan')), ('covered_fraction', True), ('covered_fraction', -1), ('covered_fraction', 1.1), ('covered_fraction', 0), ('mean_alpha', float('inf')), ('mean_alpha', False), ('mean_alpha', 0), ('mean_alpha', -1), ('intent', 'guess'), ('material', 'glass'), ('name', ''), ('name', 'bad name'), ('mask_sha256', '0' * 64)])
def test_bad_region_parameters(field, value):
    (rgb, rgba, r) = fixture()
    r[0][field] = value
    with pytest.raises(ValueError):
        call(rgb, rgba, r)

@pytest.mark.parametrize('kind', ['empty', 'wrong-shape', 'uint8', 'none'])
def test_invalid_masks(kind):
    (rgb, rgba, r) = fixture()
    r[0]['mask'] = {'empty': np.zeros((20, 30), bool), 'wrong-shape': np.ones((21, 30), bool), 'uint8': np.ones((20, 30), np.uint8), 'none': None}[kind]
    with pytest.raises(ValueError):
        call(rgb, rgba, r)

@pytest.mark.parametrize('kind', ['review', 'note', 'binding', 'candidate', 'dtype', 'regions', 'field', 'duplicate', 'overlap'])
def test_invalid_contracts(kind):
    (rgb, rgba, r) = fixture()
    kw = {}
    if kind == 'review':
        kw['reviewed'] = 1
    elif kind == 'note':
        kw['review_note'] = ' '
    elif kind == 'binding':
        kw['bindings'] = {'source': digest(rgb), 'candidate': '0' * 64}
    elif kind == 'candidate':
        rgba = rgba[:, :29]
    elif kind == 'dtype':
        rgba = rgba.astype(float)
    elif kind == 'regions':
        r = []
    elif kind == 'field':
        r[0]['extra'] = True
    elif kind == 'duplicate':
        r.append(dict(r[0]))
    elif kind == 'overlap':
        r.append(dict(r[0], name='other', intent='exclude'))
    with pytest.raises(ValueError):
        call(rgb, rgba, r, **kw)

def test_threshold_boundary_and_exclusion_equality():
    (rgb, rgba, r) = fixture()
    r[0].update(alpha_threshold=128, covered_fraction=1, mean_alpha=128 / 255)
    rgba[r[0]['mask'], 3] = 128
    assert call(rgb, rgba, r)['declared_coverage_satisfied']
    rgba[r[0]['mask'], 3] = 127
    assert not call(rgb, rgba, r)['declared_coverage_satisfied']

def test_same_intent_overlap_allowed_and_unknown_material_not_upgraded():
    (rgb, rgba, r) = fixture()
    r.append(dict(r[0], name='subpart', material='unknown'))
    out = call(rgb, rgba, r)
    assert out['declared_coverage_satisfied'] and (not out['object_completeness_verified'])

def test_four_megapixel_budget_no_automatic_resize():
    rgb = np.zeros((2001, 2000, 3), np.uint8)
    rgba = np.zeros((2001, 2000, 4), np.uint8)
    with pytest.raises(ValueError, match='4MP'):
        call(rgb, rgba, [])

def test_studio_blocks_render_without_deleting_failed_diagnostics(tmp_path):
    from PIL import Image
    import json
    from renderello.candidate_studio import run
    (rgb, rgba, r) = fixture()
    rgba[r[0]['mask'], 3] = 0
    Image.fromarray(rgb).save(tmp_path / 'source.png')
    Image.fromarray(rgba[:, :, 3]).save(tmp_path / 'alpha.png')
    Image.fromarray(rgb).save(tmp_path / 'bg.png')
    spec = dict(schema='renderello-candidate-studio-v1', source_pixels_sha256=digest(rgb), candidate_sha256={'trial': digest(rgba[:, :, 3])}, selection='trial', material='opaque', framing='complete', source_pose='front', camera='front', mask_reviewed=True, contact_reviewed=True)
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    request = dict(regions=r, bindings=dict(source=digest(rgb), candidate=digest(rgba)), reviewed=True, review_note='Source core declaration, not completeness proof.')
    run(tmp_path / 'source.png', tmp_path / 'out', {'trial': tmp_path / 'alpha.png'}, spec_path=tmp_path / 'spec.json', background=tmp_path / 'bg.png', instance_reviews={'trial': request})
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert report['render_status'] == 'BLOCKED'
    assert any((x.startswith('INSTANCE_PRESENCE_') for x in report['render_blockers']))
    assert (tmp_path / 'out/candidates/trial/alpha.png').exists()
    assert report['accepted_count'] == 0

@pytest.mark.parametrize('reviews', [{}, [], {'wrong': {}}])
def test_studio_refuses_missing_or_mismatched_review_candidates(tmp_path, reviews):
    from PIL import Image
    from renderello.candidate_studio import run
    (rgb, rgba, r) = fixture()
    Image.fromarray(rgb).save(tmp_path / 'source.png')
    Image.fromarray(rgba[:, :, 3]).save(tmp_path / 'alpha.png')
    with pytest.raises(ValueError, match='exactly every'):
        run(tmp_path / 'source.png', tmp_path / 'out', {'trial': tmp_path / 'alpha.png'}, instance_reviews=reviews)
