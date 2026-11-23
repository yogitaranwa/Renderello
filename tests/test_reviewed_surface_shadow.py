from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_surface_shadow import SurfaceShadow,supplement_surface_shadow
from renderello.candidate_harmonization import linear_over


def data():
    rgb=np.full((80,100,3),120,np.uint8);a=np.zeros((80,100),np.uint8);a[10:51,30:70]=255
    bg=np.full_like(rgb,230);shade=np.zeros(a.shape,float);shade[51:54,40:60]=.3
    p=SurfaceShadow(50,50,20.,5.,.2,digest(rgb),digest(a),digest(bg),digest(shade),'Synthetic neutral floor declaration, not physical truth.',reviewed=True)
    return rgb,a,bg,shade,p


def test_pixel_invariants_and_finite_footprint():
    rgb,a,bg,c,p=data();out,s,m=supplement_surface_shadow(rgb,a,bg,c,p)
    assert np.array_equal(out[a==255],rgb[a==255])
    assert not s[:36].any() and not s[67:].any()
    # No flat onset seam: background beside the base is already shaded above floor.
    assert s[50,20]>0 and abs(s[50,20]-s[51,20])<.01
    assert np.all(s>=c) and (s<=.6).all()
    assert (s[51,10:90]>0).all()
    base=linear_over(rgb,a,bg);assert np.array_equal(out[s==0],base[s==0])
    assert m['broad_pixels']>0 and not m['physical_relighting'] and not m['accepted']


@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'surface_kind':'wall'},
 {'center_x':True},{'center_x':-1},{'floor_row':1.5},{'floor_row':-1},
 {'sigma_x':0},{'sigma_x':257},{'sigma_y':0},{'sigma_y':33},
 {'opacity':0},{'opacity':.36},{'opacity':float('nan')},{'sigma_y':float('inf')},
 {'sigma_x':True},{'review_note':''},{'review_note':'x'*2001},
 {'source_sha256':'x'*64},{'alpha_sha256':None},{'background_sha256':'0'}, {'contact_sha256':'F'*64}])
def test_bad_policy(kw):
    with pytest.raises(ValueError):replace(data()[-1],**kw)


@pytest.mark.parametrize('kind',['rgb','alpha','background','contact'])
def test_changed_binding(kind):
    v=list(data());idx={'rgb':0,'alpha':1,'background':2,'contact':3}[kind];v[idx]=v[idx].copy();v[idx].flat[0]+=1
    with pytest.raises(ValueError):supplement_surface_shadow(*v)


@pytest.mark.parametrize('kind',['integer','shape','nan','negative','high','on_fg','no_attach','anchor','center','wrong_policy'])
def test_bad_arrays_or_geometry(kind):
    rgb,a,bg,c,p=data()
    if kind=='integer':c=c.astype(np.uint8)
    elif kind=='shape':c=c[:-1]
    elif kind=='nan':c[60,60]=np.nan
    elif kind=='negative':c[60,60]=-.1
    elif kind=='high':c[60,60]=.6
    elif kind=='on_fg':c[a==255]=.1;p=replace(p,contact_sha256=digest(c))
    elif kind=='no_attach':c[:]=0;p=replace(p,contact_sha256=digest(c))
    elif kind=='anchor':p=replace(p,floor_row=79)
    elif kind=='center':p=replace(p,center_x=100)
    elif kind=='wrong_policy':p=None
    with pytest.raises(ValueError):supplement_surface_shadow(rgb,a,bg,c,p)


def test_no_inputs_mutated():
    v=data();before=[a.copy() for a in v[:4]];supplement_surface_shadow(*v)
    assert all(np.array_equal(a,b) for a,b in zip(v[:4],before))
