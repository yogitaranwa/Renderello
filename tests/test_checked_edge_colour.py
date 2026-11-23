import numpy as np
import pytest
from renderello.checked_research.edge_colour import estimate_fractional_edge


def pair():
    rgb=np.full((8,10,3),128,np.uint8)
    a=np.zeros((8,10),np.uint8);a[:,2:8]=255;a[:,2]=100;a[:,7]=200
    return rgb,a


def fixed(image,alpha,**kwargs):
    return np.full_like(image,.25),np.full_like(image,.75)


def test_only_fractional_changes_and_inputs_not_mutated():
    rgb,a=pair();before=rgb.copy();mask=a.copy()
    out,m=estimate_fractional_edge(rgb,a,source_alpha_pair_reviewed=True,estimator=fixed)
    assert np.array_equal(rgb,before) and np.array_equal(a,mask)
    assert np.all(out[(a>0)&(a<255)]==64)
    assert np.array_equal(out[(a==0)|(a==255)],rgb[(a==0)|(a==255)])
    assert m['publication_accepted'] is False and m['boundary_accuracy'] is None
    assert m['opaque_rgb_exact'] and not m['source_rgb_fully_exact']
    assert m['estimator_uses_internal_pyramid'] and m['injected_estimator']


def test_gate():
    rgb,a=pair()
    with pytest.raises(ValueError,match='pairing review'):estimate_fractional_edge(rgb,a,estimator=fixed)


@pytest.mark.parametrize('bad',['rgbfloat','rgbgray','alphafloat','alphashape','empty','opaque','binary','noopaque','nobg','huge','tiny'])
def test_bad_inputs(bad):
    rgb,a=pair()
    if bad=='rgbfloat':rgb=rgb.astype(float)
    if bad=='rgbgray':rgb=rgb[:,:,0]
    if bad=='alphafloat':a=a.astype(float)
    if bad=='alphashape':a=a[:-1]
    if bad=='empty':a[:]=0
    if bad=='opaque':a[:]=255
    if bad=='binary':a[a>0]=255
    if bad=='noopaque':a[a==255]=200
    if bad=='nobg':a[a==0]=128
    if bad=='huge':rgb=np.zeros((1001,1000,3),np.uint8);a=np.zeros((1001,1000),np.uint8)
    if bad=='tiny':rgb=rgb[:1];a=a[:1]
    with pytest.raises(ValueError):estimate_fractional_edge(rgb,a,source_alpha_pair_reviewed=True,estimator=fixed)


@pytest.mark.parametrize('bad',['nan','inf','shape','integer','negative','over','none'])
def test_bad_estimator_outputs(bad):
    rgb,a=pair()
    def estimator(image,alpha,**kwargs):
        f,b=fixed(image,alpha)
        if bad=='nan':f[0,0,0]=np.nan
        if bad=='inf':b[0,0,0]=np.inf
        if bad=='shape':f=f[:-1]
        if bad=='integer':f=f.astype(int)
        if bad=='negative':b[0,0,0]=-.1
        if bad=='over':f[0,0,0]=1.1
        if bad=='none':b=None
        return f,b
    with pytest.raises(ValueError):estimate_fractional_edge(rgb,a,source_alpha_pair_reviewed=True,estimator=estimator)
