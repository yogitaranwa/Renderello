import copy
from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.matte_change_review import ChangeBudget
from renderello.reviewed_candidate_gate import review_candidate


def case():
    rgb=np.full((24,32,3),127,np.uint8)
    before=np.zeros((24,32),np.uint8);before[5:20,5:25]=255
    rgba=np.dstack([rgb,before]);allowed=np.zeros(before.shape,bool);allowed[5:8,5:25]=True
    protected=np.zeros(before.shape,bool);protected[10:15,10:20]=True
    return dict(rgb=rgb,before=before,rgba=rgba,allowed=allowed,protected=protected)


def run(c, **kw):
    names=dict(source='rgb',before='before',candidate='rgba',allowed='allowed',protected='protected')
    args=dict(bindings={n:digest(c[k])for n,k in names.items()},reviewed=True,
              material='opaque',framing='complete',review_note='Synthetic exact region fixture, not a photograph.')
    args.update(kw);return review_candidate(**c,**args)


def test_identity_not_quality_or_selection():
    c=case();snap=copy.deepcopy(c);r,delta=run(c)
    assert r['intervention_constraints_satisfied'] and not r['photo_accepted']
    assert not r['candidate_selected'] and not r['output_repaired'] and not delta.any()
    assert r['baseline_accuracy']==r['candidate_accuracy']=='NOT_VERIFIED'
    assert all(np.array_equal(c[k],snap[k])for k in c)


def test_local_fractional_candidate_within_bounds_is_not_promoted():
    c=case();c['rgba'][5,6,3]=254;r,_=run(c)
    assert r['changed_alpha_pixels']==1 and r['intervention_constraints_satisfied']
    assert not r['photo_accepted'] and not r['default_changed']


def test_single_soft_change_outside_region_is_blocked():
    c=case();c['rgba'][8,8,3]=254;r,_=run(c)
    assert r['changed_outside_edit_pixels']==1
    assert 'ALPHA_CHANGED_OUTSIDE_REVIEWED_EDIT_REGION'in r['blockers']


def test_protected_soft_change_is_blocked_even_if_globally_tiny():
    c=case();c['rgba'][11,12,3]=254;r,_=run(c)
    assert r['changed_protected_alpha_pixels']==1 and not r['intervention_constraints_satisfied']
    assert 'PROTECTED_ALPHA_CHANGED'in r['blockers']


@pytest.mark.parametrize('channel',[0,1,2])
@pytest.mark.parametrize('point',[(0,0),(11,12),(5,6)])
def test_rgb_change_rejected_even_under_zero_alpha(channel,point):
    c=case();c['rgba'][point[0],point[1],channel]=126;r,_=run(c)
    assert r['changed_source_rgb_pixels']==1 and 'CANDIDATE_SOURCE_RGB_CHANGED'in r['blockers']


def test_topology_can_block_inside_allowed_region():
    c=case();c['before'][5,29]=255;c['rgba'][5,29,3]=0;c['allowed'][5,29]=True;r,_=run(c)
    assert r['changed_outside_edit_pixels']==0
    assert 'TOPOLOGY_INITIAL_PART_DAMAGED_OR_REMOVED'in r['blockers']


@pytest.mark.parametrize('name',['rgb','before','rgba','allowed','protected'])
def test_stale_binding(name):
    c=case();names=dict(source='rgb',before='before',candidate='rgba',allowed='allowed',protected='protected')
    binds={n:digest(c[k])for n,k in names.items()};c[name]=c[name].copy();c[name].flat[0]=not c[name].flat[0] if c[name].dtype==bool else 12
    with pytest.raises(ValueError):run(c,bindings=binds)


@pytest.mark.parametrize('kw',[dict(reviewed=False),dict(reviewed=1),dict(material='glass'),dict(material='unknown'),dict(framing='cropped'),dict(framing=None),dict(review_note=''),dict(review_note=' '*10),dict(review_note='a'*2001),dict(review_note=1),dict(bindings={}),dict(bindings=[]),dict(budget={})])
def test_missing_or_invalid_review(kw):
    with pytest.raises(ValueError):run(case(),**kw)


@pytest.mark.parametrize('name',['allowed','protected'])
@pytest.mark.parametrize('change',['empty','float','wrongshape','list'])
def test_region_validation(name,change):
    c=case()
    c[name]={'empty':np.zeros(c[name].shape,bool),'float':c[name].astype(float),'wrongshape':np.ones((4,4),bool),'list':c[name].tolist()}[change]
    with pytest.raises((ValueError,AttributeError)):run(c)


def test_regions_cannot_overlap():
    c=case();c['allowed']|=c['protected']
    with pytest.raises(ValueError,match='overlap'):run(c)


def test_protecting_only_empty_background_is_refused():
    c=case();c['protected'][:]=False;c['protected'][0,0]=True
    with pytest.raises(ValueError,match='foreground'):run(c)


@pytest.mark.parametrize('change',['float','rgb','wrongshape'])
def test_candidate_validation(change):
    c=case();c['rgba']={'float':c['rgba'].astype(float),'rgb':c['rgb'],'wrongshape':np.zeros((3,3,4),np.uint8)}[change]
    with pytest.raises(ValueError):run(c)


def test_exact_noncontiguous_inputs_supported_without_mutation():
    c={k:v[:,::-1].copy()[:,::-1]for k,v in case().items()};r,_=run(c)
    assert r['intervention_constraints_satisfied']


def test_returned_bindings_do_not_alias_caller_dict():
    c=case();names=dict(source='rgb',before='before',candidate='rgba',allowed='allowed',protected='protected');binds={n:digest(c[k])for n,k in names.items()};r,_=run(c,bindings=binds);r['bindings']['source']='changed';assert binds['source']==digest(c['rgb'])
