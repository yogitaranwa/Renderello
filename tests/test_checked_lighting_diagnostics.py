import numpy as np
import pytest
from renderello.checked_research.lighting_diagnostics import paired_horizontal_gain as diagnostic

def inputs():
    mask=np.zeros((20,16),bool);mask[2:18,2:14]=True
    l=np.tile(np.linspace(1.4,.6,16),(20,1));r=l[:,::-1].copy()
    return l,r,mask

def test_directional_fixture_changes_sign():
    l,r,m=inputs();x=diagnostic(l,r,m,aligned_pair_reviewed=True)
    assert x['rows'][0]['left_minus_right']>0 and x['rows'][1]['left_minus_right']<0
    assert x['contrast_change_right_minus_left']<0
    assert x['independent_direction_accuracy'] is None and not x['physical_lighting_established']
    assert not x['publication_accepted']

def test_identity_pair_no_response():
    l,r,m=inputs();x=diagnostic(l,l,m,aligned_pair_reviewed=True)
    assert x['contrast_change_right_minus_left']==0 and x['gain_mean_abs_difference']==0

@pytest.mark.parametrize('approval',[False,None,1,'yes'])
def test_needs_explicit_review(approval):
    with pytest.raises(ValueError):diagnostic(*inputs(),aligned_pair_reviewed=approval)

@pytest.mark.parametrize('kind',['nan','inf','zero','negative','int','complex','shape'])
def test_invalid_gains(kind):
    l,r,m=inputs()
    if kind=='int':l=l.astype(int)
    elif kind=='complex':l=l.astype(complex)
    elif kind=='shape':l=l[:-1]
    else:l[0,0]={'nan':np.nan,'inf':np.inf,'zero':0,'negative':-1}[kind]
    with pytest.raises(ValueError):diagnostic(l,r,m,aligned_pair_reviewed=True)

@pytest.mark.parametrize('kind',['empty','one_column','uint8','tiny','shape'])
def test_invalid_support(kind):
    l,r,m=inputs()
    if kind=='empty':m[:]=False
    elif kind=='one_column':m[:]=False;m[:,4]=True
    elif kind=='uint8':m=m.astype('uint8')
    elif kind=='tiny':m[:]=False;m[1:3,1:3]=True
    elif kind=='shape':m=m[:-1]
    with pytest.raises(ValueError):diagnostic(l,r,m,aligned_pair_reviewed=True)
