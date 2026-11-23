import numpy as np
import pytest
from renderello.checked_research.contact_camera import SceneSpec, layout, composite, shadow_map, smooth_scene, linear, srgb


def sample():
    rgb=np.full((60,80,3),[140,90,20],np.uint8)
    a=np.zeros((60,80),np.uint8);a[5:55,25:60]=255;a[5:55,24]=127
    return rgb,a,np.full((48,48,3),205,np.uint8)


@pytest.mark.parametrize('camera',['front','elevated','flatlay'])
@pytest.mark.parametrize('light',['left','right','diffuse'])
def test_native_core(camera,light):
    rgb,a,b=sample();out,bg,shade,m=composite(rgb,a,b,SceneSpec(camera,light))
    l,t,r,bot=m['sourceCrop'];x,y=m['nativePlacement'];sub=out[y:y+bot-t,x:x+r-l]
    assert np.array_equal(sub[a[t:bot,l:r]==255],rgb[t:bot,l:r][a[t:bot,l:r]==255])
    assert m['changedOpaquePixels']==0 and not m['foregroundResampled']
    assert np.isfinite(shade).all() and 0<=shade.min()<=shade.max()<=.68


def test_contact_touches_lower_contour():
    _,a,_=sample();box,size,pos=layout(a);l,t,r,b=box;aa=a[t:b,l:r]
    sh=shadow_map(aa,size[::-1],pos,SceneSpec('front'))
    x,y=pos;assert sh[y+aa.shape[0],x+10]>.1


def test_crop_removes_margin_without_scaling():
    _,a,_=sample();box,size,pos=layout(a)
    assert box==(24,5,60,55) and size==(256,256)


def test_review_flags_fail_closed():
    rgb,a,b=sample();*_,m=composite(rgb,a,b,SceneSpec('front',transparent=True,cropped=True,multiobject=True))
    assert len(m['unresolvedFlags'])==4 and m['commercialAcceptance'] is None


def test_no_false_optics_or_relight():
    rgb,a,b=sample();*_,m=composite(rgb,a,b,SceneSpec('front'))
    assert not m['glassOpticsRecovered'] and not m['sourceRelit']


def test_rgb_roundtrip():
    a=np.arange(256,dtype=np.uint8);assert np.array_equal(srgb(linear(a)),a)


@pytest.mark.parametrize('bad',[np.zeros((10,10),np.uint8),np.ones((2,2),float),np.zeros((1,4),np.uint8)])
def test_bad_alpha(bad):
    with pytest.raises(ValueError):layout(bad)


@pytest.mark.parametrize('bad',['side','unknown',''])
def test_bad_camera(bad):
    with pytest.raises(ValueError):SceneSpec(bad)


def test_smooth_field_has_no_local_prop_edges():
    b=np.full((64,64,3),205,np.uint8);b[20:40,20:40]=20
    out=smooth_scene(b,(100,100),SceneSpec('front'))
    assert np.max(np.abs(np.diff(out.astype(float),axis=0)))<3
    assert np.max(np.abs(np.diff(out.astype(float),axis=1)))<3


def test_upright_contact_peak_is_first_exposed_row():
    a=np.full((150,90),255,np.uint8)
    sh=shadow_map(a,(250,250),(70,35),SceneSpec('front'))
    profile=sh[185:215,115]
    assert profile[0]>.5
    assert np.argmax(profile)==0
    assert np.all(np.diff(profile)<=1e-10)
    assert profile[12]<profile[0]*.08


def test_upright_contact_follows_reviewed_lower_span():
    a=np.zeros((100,120),np.uint8)
    a[10:90,20:100]=255
    a[89:94,40:80]=255
    sh=shadow_map(a,(200,220),(40,30),SceneSpec('front'))
    assert sh[124,100]>.5
    assert sh[130,100]<sh[124,100]
    assert sh[124,100]>sh[124,50]*3


def test_transparent_case_never_certifies_correct_optics():
    rgb,a,b=sample()
    *_,m=composite(rgb,a,b,SceneSpec('front',transparent=True,support_reviewed=True))
    assert 'transparent-material-optics-not-recovered' in m['unresolvedFlags']
    assert not m['glassOpticsRecovered'] and m['commercialAcceptance'] is None
