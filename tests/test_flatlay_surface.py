import numpy as np
from PIL import Image
import pytest
from renderello.compose import surface_contact_shadow,composite_with_scene,finish_scene
from renderello.iclight_gpu import FlatlaySurfaceCallback
from renderello.schemas import CutoutResult,Aspect
from renderello.templates import get_template


def test_flatlay_strips_all_source_conditioning(tmp_path):
    a=np.zeros((100,100,4),np.uint8);a[20:90,20:80]=[200,0,20,255]
    p=tmp_path/'p.png';Image.fromarray(a).save(p);co=CutoutResult(p,p,(20,20,60,70),'test',0.)
    class Capture:
        flatlay_surface=True
        def __call__(self,a,p):self.a=a;self.p=p;return np.full_like(a,190)
    c=Capture();im,d=composite_with_scene(co,get_template('flat_lay'),Aspect.SQUARE,0,c)
    assert np.unique(c.a).tolist()==[176]
    assert '90 degrees' in c.p and 'no horizon' in c.p
    assert d['paste_bit_exact'] and d['shadow_style']=='surface contact'
    assert d['scene_mode']=='flatlay_surface_experimental'


def test_surface_contact_is_full_silhouette_and_bounded():
    m=np.full((30,20),255,np.uint8);s=surface_contact_shadow((100,100),(30,20),m)
    assert s.shape==(100,100) and s.max()<=65
    assert s[24:48,34:48].mean()>30
    assert s[:10].max()==0
    with pytest.raises(ValueError):surface_contact_shadow((100,100),(90,20),m)


def test_surface_finish_preserves_product():
    s=np.full((100,100,3),190,np.uint8);c=np.zeros((30,30,4),np.uint8);c[:]=[123,87,66,255]
    out,_=finish_scene(s,c,(30,20),shadow_style='surface contact')
    assert np.array_equal(out[20:50,30:60],c[...,:3])
    assert np.any(out[50:55,30:60]<190)


def test_callback_contract():
    class M:
        def relight(self,a,p,**k):self.k=k;return a,0.
    m=M();c=FlatlaySurfaceCallback(m,7);a=np.full((20,20,3),176,np.uint8)
    assert np.array_equal(c(a,'flat surface'),a)and m.k['seed']==7
    assert 'wall' in m.k['n_prompt']
    with pytest.raises(ValueError):c(a.astype(float),'p')


def test_flatlay_alpha_center_not_baseline():
    from renderello.compose import flatlay_center
    m=np.zeros((200,160),np.uint8);m[10:170,20:100]=255
    x,y=flatlay_center(m,(500,500))
    assert abs((x+59.5)-250)<=.5 and abs((y+89.5)-250)<=.5
    with pytest.raises(ValueError):flatlay_center(np.zeros((20,20),np.uint8),(100,100))
    with pytest.raises(ValueError):flatlay_center(m,(100,100))


def test_materials_are_explicit_and_prompt_distinct():
    class M:
        def relight(self,a,p,**k):self.prompt=p;return a,0.
    m=M();a=np.full((10,10,3),176,np.uint8);prompts=[]
    for material in ['linen','slate','paper']:
        FlatlaySurfaceCallback(m,0,material)(a,'ignored generic template')
        prompts.append(m.prompt)
    assert len(set(prompts))==3
    with pytest.raises(ValueError):FlatlaySurfaceCallback(m,0,'unknown')


def test_quiet_ring_never_claims_visual_acceptance(tmp_path):
    a=np.zeros((100,100,4),np.uint8);a[20:90,20:80]=[200,0,20,255]
    p=tmp_path/'p.png';Image.fromarray(a).save(p);co=CutoutResult(p,p,(20,20,60,70),'test',0.)
    class Quiet:
        flatlay_surface=True
        def __call__(self,a,p):return np.full_like(a,190)
    _,d=composite_with_scene(co,get_template('flat_lay'),Aspect.SQUARE,0,Quiet())
    assert d['ring_alert'] is False
    assert d['needs_eye_review'] is True and d['publication_status']=='NOT_EVALUATED'
