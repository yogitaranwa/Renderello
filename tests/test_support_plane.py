import numpy as np
from PIL import Image
import pytest
from renderello.compose import support_plane_canvas,composite_with_scene
from renderello.iclight_gpu import SupportPlaneCallback
from renderello.schemas import Aspect,CutoutResult
from renderello.templates import get_template


def test_plane_contains_only_two_neutral_values():
    a,h=support_plane_canvas((100,200,3),80);assert h==62
    assert a.shape==(100,200,3) and a.dtype==np.uint8
    assert np.all(a[:62]==127) and np.all(a[62:]==176)


def test_bad_support_rejected():
    for y in [-1,100]:
        with pytest.raises(ValueError):support_plane_canvas((100,200,3),y)


def test_compositor_strips_source_shape_and_rgb(tmp_path):
    a=np.zeros((150,150,4),np.uint8);a[30:100,40:110,:3]=(255,0,30);a[30:100,40:110,3]=255
    p=tmp_path/'p.png';Image.fromarray(a).save(p);co=CutoutResult(p,p,(40,30,70,70),'test',.2)
    class Capture:
        support_plane=True
        def __call__(self,canvas,prompt):
            self.canvas=canvas;self.prompt=prompt;return np.full_like(canvas,180)
    cb=Capture();im,d=composite_with_scene(co,get_template('clean_studio'),Aspect.PORTRAIT,0,cb,shadow_style='anchored soft')
    assert set(np.unique(cb.canvas))=={127,176}
    assert 'empty bare level tabletop' in cb.prompt
    assert d['scene_mode']=='support_plane_experimental' and d['paste_bit_exact']


def test_callback_uses_free_existing_model_contract():
    class Model:
        def relight(self,canvas,prompt,**kwargs):self.k=kwargs;return canvas,1
    m=Model();cb=SupportPlaneCallback(m,seed=7);a=np.full((100,80,3),127,np.uint8)
    assert np.array_equal(cb(a,'empty table'),a);assert m.k['seed']==7 and 'doorway'in m.k['n_prompt']
    with pytest.raises(ValueError):cb(a.astype(float),'empty table')
