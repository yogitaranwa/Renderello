import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.contact_spans import ContactPolicy,ContactSpan
from renderello.authored_platform_contact import platform_scene,compose_platform_contact

@pytest.fixture
def data():
    rgb=np.full((50,60,3),120,np.uint8);a=np.zeros((50,60),np.uint8);a[5:31,15:45]=255
    rgb[a==255]=(100,70,40)
    return rgb,a

def call(data,**kw):
    rgb,a=data;p=ContactPolicy('front','opaque','complete','front',digest(rgb),digest(a),reviewed=True,length=24,lateral_sigma=3)
    args=dict(style='clay_plinth',spans=[ContactSpan('base',15,45,30,30)],policy=p,
        translation=[470,845],native_floor_row=30)
    args.update(kw);return compose_platform_contact(rgb,a,**args)

@pytest.mark.parametrize('style',['clay_plinth','blush_steps'])
def test_actual_surface_attachment_and_exact_source(data,style):
    before=[x.copy() for x in data];out,shade,bg,plane,m=call(data,style=style)
    assert m['attachments'][0]['columns']==30 and m['shadow_pixels_outside_surface']==0
    assert np.all(shade[876,485:515]>0) and not shade[~plane].any()
    rgb,a=data;view=out[845:895,470:530];assert np.array_equal(view[a==255],rgb[a==255])
    assert np.array_equal(out[~plane & (np.pad(a,((845,129),(470,494)))==0)],bg[~plane & (np.pad(a,((845,129),(470,494)))==0)])
    assert m['opaque_source_pixels_exact'] and not m['accepted'] and not m['physical_relighting']
    assert all(np.array_equal(a,b) for a,b in zip(data,before))

@pytest.mark.parametrize('style',['clay_plinth','blush_steps'])
def test_background_surface_deterministic(style):
    a,m,r=platform_scene(style);b,n,s=platform_scene(style)
    assert np.array_equal(a,b) and np.array_equal(m,n) and r==s
    assert r['surface_sha256']==digest(m) and m[875,500] and not m[910,500]

@pytest.mark.parametrize('kwargs',[{'style':'porcelain_sweep'},{'seed':True},{'seed':-1},{'seed':2**32},{'seed':1.5}])
def test_bad_design(kwargs):
    with pytest.raises(ValueError):platform_scene(**dict(style='clay_plinth',**kwargs) if 'style' not in kwargs else kwargs)

@pytest.mark.parametrize('kw',[{'translation':[-1,845]},{'translation':[470.0,845]},
 {'translation':[True,845]},{'translation':[470]},{'translation':None},{'translation':[1000,845]},
 {'native_floor_row':True},{'native_floor_row':-1},{'native_floor_row':50},{'native_floor_row':10},
 {'support_tolerance':True},{'support_tolerance':5},{'support_tolerance':-1},
 {'spans':[]},{'spans':['base']},{'spans':[ContactSpan('outside',15,100,30,30)]},
 {'translation':[470,810]},{'translation':[50,845]}, {'style':'noir_halo'}])
def test_invalid_placement_not_silently_fixed(data,kw):
    with pytest.raises(ValueError):call(data,**kw)

@pytest.mark.parametrize('changes',[{'reviewed':False},{'source_sha256':'0'*64},{'alpha_sha256':'0'*64},
 {'material':'transparent'},{'framing':'cropped'},{'camera':'elevated','source_pose':'elevated'},
 {'camera':'elevated'},{'source_pose':'suspended'}])
def test_bad_source_review(data,changes):
    rgb,a=data;p=dict(camera='front',material='opaque',framing='complete',source_pose='front',source_sha256=digest(rgb),alpha_sha256=digest(a),reviewed=True);p.update(changes)
    with pytest.raises(ValueError):call(data,policy=ContactPolicy(**p))

def test_frame_touching_mask_refused(data):
    rgb,a=data;a[0,20]=255
    with pytest.raises(ValueError):call(data)

def test_partial_alpha_remains_fractional_not_repainted(data):
    rgb,a=data;a[12:20,14]=128
    out,shade,bg,plane,m=call(data)
    assert not np.array_equal(out[857:865,484],rgb[12:20,14]) and np.array_equal(rgb,data[0])
    assert m['source_rgb_repainted'] is False
