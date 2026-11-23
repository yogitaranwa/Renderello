"""Native topology regressions. Passing does not certify mask correctness."""
from dataclasses import replace
import json
import cv2
import numpy as np
import pytest
from PIL import Image
from renderello.matte_change_review import ChangeBudget, audit_trimap, review_change
from renderello.candidate_diagnostics import digest


def mask():
    a = np.zeros((80,100), np.uint8)
    a[15:60,15:60] = 255
    a[25:40,25:40] = 0
    a[45:50,75:80] = 255
    return a


def test_identity_preserves_parts_and_handle_hole():
    a = mask(); before = a.copy()
    r, view = review_change(a,a)
    assert r['initial_parts'] == r['revised_parts'] == 2
    assert r['original_holes'] == 1 and r['filled_hole_count'] == 0
    assert r['within_configured_change_budget'] and not r['automatic_acceptance']
    assert r['baseline_accuracy'] == 'NOT_VERIFIED'
    assert not view.any() and np.array_equal(a,before)
    assert r['before_sha256'] == digest(a)


def test_small_accessory_removal_not_hidden_by_large_object():
    a = mask(); b = a.copy(); b[45:50,75:80] = 0
    r, view = review_change(a,b)
    assert r['removed_fraction'] < .1
    assert r['damaged_part_count'] == 1
    assert 'INITIAL_PART_DAMAGED_OR_REMOVED' in r['blockers']
    assert np.all(view[45:50,75:80,2] == 255)
    assert not view[:,:,0].any()


def test_handle_hole_fill_is_not_a_matte_improvement():
    a = mask(); b = a.copy(); b[25:40,25:40] = 255
    r,_ = review_change(a,b)
    assert r['filled_hole_count'] == 1
    assert 'INITIAL_HOLE_FILLED' in r['blockers']
    assert r['holes'][0]['filled_fraction'] == 1


def test_bridge_between_initial_parts_flagged():
    a = mask(); b = a.copy(); b[46:49,59:76] = 255
    r,_ = review_change(a,b)
    assert r['bridge_count'] == 1
    assert r['revised_parts'] == 1
    assert 'REVISED_COMPONENT_BRIDGES_INITIAL_PARTS' in r['blockers']


def test_new_island_not_discarded():
    a = mask(); b = a.copy(); b[3:5,3:5] = 255
    r,_ = review_change(a,b)
    assert r['new_component_count'] == 1
    assert 'NEW_DISCONNECTED_FOREGROUND' in r['blockers']
    assert b[3:5,3:5].sum() == 4*255


def test_small_boundary_change_can_pass_budget_but_not_quality():
    a = mask(); b = a.copy(); b[20:25,60] = 255
    r,_ = review_change(a,b)
    assert r['within_configured_change_budget']
    assert r['added_hard_pixels'] == 5
    assert not r['automatic_acceptance']


def test_soft_alpha_delta_is_separate_from_hard_topology():
    a = mask(); b = a.copy(); b[10:15,15:60] = 70
    r,view = review_change(a,b)
    assert r['added_hard_pixels'] == 0
    assert r['changed_alpha_pixels'] == 225
    assert r['maximum_alpha_delta'] == 70
    assert np.all(view[10:15,15:60,0] == 70)


def test_global_alpha_change_budget():
    a = np.full((20,20), 129,np.uint8); b = np.full_like(a,250)
    r,_ = review_change(a,b)
    assert 'ALPHA_CHANGE_EXCEEDS_BUDGET' in r['blockers']
    assert r['initial_parts'] == 1


@pytest.mark.parametrize('value',[0,255])
def test_empty_full_do_not_crash(value):
    a = np.full((10,10),value,np.uint8); r,_ = review_change(a,a)
    assert isinstance(r['blockers'],list)
    if value == 0: assert 'EMPTY_INITIAL_HARD_MASK' in r['blockers']


@pytest.mark.parametrize('kw',[dict(threshold=True),dict(threshold=0),dict(threshold=256),
    dict(detail_limit=True),dict(detail_limit=0),dict(detail_limit=257),
    dict(maximum_hole_fill=float('nan')),dict(maximum_added_fraction=float('inf')),
    dict(minimum_part_retained=-.1),dict(maximum_removed_fraction=1.1),
    dict(maximum_alpha_mean_delta=True),dict(maximum_alpha_mean_delta='1')])
def test_invalid_budgets(kw):
    with pytest.raises(ValueError):ChangeBudget(**kw)


@pytest.mark.parametrize('bad',[None,np.zeros((3,3)),np.zeros((3,3,1),np.uint8),
    np.zeros((2,2),np.uint8),np.zeros((3,4),np.uint8)])
def test_bad_revision_arrays(bad):
    a = np.zeros((3,3),np.uint8)
    with pytest.raises(ValueError):review_change(a,bad)


def test_bad_initial_and_untyped_budget():
    with pytest.raises(ValueError):review_change(None,None)
    with pytest.raises(ValueError):review_change(mask(),mask(),{})


def test_detail_limit_does_not_drop_mask_or_aggregate_counts():
    a = np.zeros((80,80),np.uint8);a[2:78:3,2:78:3] = 255
    r,_ = review_change(a,a,ChangeBudget(detail_limit=4))
    assert r['initial_parts'] == 26*26
    assert len(r['parts']) == 4
    assert r['detail_truncated']
    assert 'TOPOLOGY_DETAIL_LIMIT_REQUIRES_REVIEW' in r['blockers']


def test_diagonal_connectivity_and_canvas_edge_not_hole():
    a = np.zeros((10,10),np.uint8);a[3,3] = a[4,4] = 255
    r,_ = review_change(a,a)
    assert r['initial_parts'] == 1 and r['original_holes'] == 0


def test_trimap_seed_conflict_is_review_not_truth():
    a = mask(); tri = np.full_like(a,128);tri[0,0] = 0;tri[30,30] = 255
    r, conflicts = audit_trimap(a,tri)
    assert r['contradictory_seed_pixels'] == 1
    assert conflicts[30,30]
    assert r['unseeded_part_count'] == 2
    assert r['seed_semantics'] == 'NOT_VERIFIED'


def test_eroded_heuristic_can_lose_thin_component_seeds():
    from renderello.matting import make_trimap
    a = mask(); a[45:50,75:80] = 0; a[45:47,75:80] = 255
    tri = make_trimap(a,2)
    r,_ = audit_trimap(a,tri)
    assert r['unseeded_part_count'] == 1
    assert 'INITIAL_PART_WITHOUT_FOREGROUND_SEED' in r['blockers']


@pytest.mark.parametrize('value',[0,128,255])
def test_missing_trimap_seed_classes(value):
    a = mask();tri = np.full_like(a,value)
    r,_ = audit_trimap(a,tri)
    assert r['blockers']


@pytest.mark.parametrize('limit',[True,0,4_000_001,1.5])
def test_invalid_unknown_budget(limit):
    with pytest.raises(ValueError):audit_trimap(mask(),mask(),unknown_limit=limit)


def test_unknown_budget_and_invalid_levels():
    a = mask();tri = np.full_like(a,128);tri[0,0] = 0;tri[20,20] = 255
    r,_ = audit_trimap(a,tri,unknown_limit=1)
    assert 'UNKNOWN_BAND_EXCEEDS_BUDGET' in r['blockers']
    tri[5,5] = 64
    with pytest.raises(ValueError):audit_trimap(a,tri)


def test_pipeline_records_audits_and_blocks_refined_damaged_part(tmp_path,monkeypatch):
    from renderello.candidate_studio import run
    import renderello.matting
    a = mask();rgb = np.full((*a.shape,3),160,np.uint8)
    tri = np.where(a == 255,255,0).astype(np.uint8)
    for name,data in [('source',rgb),('alpha',a),('trimap',tri)]:Image.fromarray(data).save(tmp_path/(name+'.png'))
    def mocked_worker(source,initial,out,**kw):
        # This tests audit propagation, not a successful model inference.
        out.mkdir();b = a.copy();b[45:50,75:80] = 0
        Image.fromarray(b).save(out/'refined-alpha.png')
    monkeypatch.setattr(renderello.matting,'guarded_file',mocked_worker)
    r = run(tmp_path/'source.png',tmp_path/'out',{'initial':tmp_path/'alpha.png'},
        matting_candidate='initial',trimap=tmp_path/'trimap.png',matting_method='guided')
    assert 'initial-refined' in r['refinement_reviews']
    assert any('INITIAL_PART_DAMAGED_OR_REMOVED' in x for x in r['candidates']['initial-refined']['render_blockers'])
    assert (tmp_path/'out/matte-alpha-change.png').exists()
    assert (tmp_path/'out/trimap-review.json').exists()
    assert r['accepted_count'] == 0


def test_real_guarded_worker_outside_repository(tmp_path,monkeypatch):
    from renderello.matting import guarded_file,make_trimap
    a = mask();rgb = np.full((*a.shape,3),160,np.uint8);tri = make_trimap(a,2)
    for name,data in [('source',rgb),('alpha',a),('trimap',tri)]:Image.fromarray(data).save(tmp_path/(name+'.png'))
    monkeypatch.chdir(tmp_path)
    r = guarded_file(tmp_path/'source.png',tmp_path/'alpha.png',tmp_path/'out',trimap_path=tmp_path/'trimap.png',method='guided')
    assert r['source_rgb_unchanged']
    assert (tmp_path/'out/refined-alpha.png').exists()
