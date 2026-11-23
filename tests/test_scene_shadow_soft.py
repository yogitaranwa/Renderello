import numpy as np
import pytest
from renderello.scene_shadow import projected_soft_shadow,attenuate_linear
from renderello.scene_plate import support_shadow

@pytest.fixture
def alpha():
 a=np.zeros((80,40),np.uint8);a[5:78,8:32]=255;return a

def test_zero_cast_identical_contact(alpha):
 assert np.array_equal(projected_soft_shadow(alpha,(180,160),(60,30)),support_shadow(alpha,(180,160),(60,30)))

def test_projected_shadow_finite_bounded(alpha):
 s=projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(60,20))
 assert s.shape==(160,180) and np.isfinite(s).all() and s.min()>=0 and s.max()<=.45

def test_cast_goes_declared_direction(alpha):
 a=projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(60,20))
 b=projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(-60,20))
 base=support_shadow(alpha,(180,160),(60,30))
 assert (a-base)[:,100:].sum()>(b-base)[:,100:].sum()

@pytest.mark.parametrize('v',[float('nan'),float('inf'),0,4])
def test_invalid_softness(alpha,v):
 with pytest.raises(ValueError):projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(20,10),softness=v)

@pytest.mark.parametrize('v',[(float('nan'),1),(181,1),(1,161)])
def test_invalid_cast(alpha,v):
 with pytest.raises(ValueError):projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=v)

def test_linear_zero_identity():
 rng=np.random.default_rng(7);a=rng.integers(0,256,(20,20,3),dtype=np.uint8)
 assert np.array_equal(attenuate_linear(a,np.zeros((20,20))),a)

def test_linear_black_and_monotonic():
 a=np.full((4,4,3),180,np.uint8)
 low=attenuate_linear(a,np.full((4,4),.2));high=attenuate_linear(a,np.full((4,4),.6))
 assert (high<low).all() and (low<a).all()
 assert not attenuate_linear(a,np.ones((4,4))).any()

@pytest.mark.parametrize('v',[float('nan'),-1,1.01])
def test_invalid_shadow_values(v):
 with pytest.raises(ValueError):attenuate_linear(np.zeros((4,4,3),np.uint8),np.full((4,4),v))

def test_softness_changes_cast_not_contact(alpha):
 a=projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(60,20),softness=.5)
 b=projected_soft_shadow(alpha,(180,160),(60,30),cast_offset=(60,20),softness=2)
 assert not np.array_equal(a,b)
 assert np.array_equal(projected_soft_shadow(alpha,(180,160),(60,30),softness=.5),projected_soft_shadow(alpha,(180,160),(60,30),softness=2))

def test_optin_composition_keeps_source_clips_plane_and_legacy_default(tmp_path):
 from PIL import Image,ImageDraw
 from renderello.reviewed_source import prepare_reviewed_source
 from renderello.scene_plate import compose_plate
 src=Image.new('RGB',(80,100),'white');a=Image.new('L',src.size,0)
 ImageDraw.Draw(src).rectangle((20,10,60,90),fill=(25,80,140));ImageDraw.Draw(a).rectangle((20,10,60,90),fill=255)
 src.save(tmp_path/'src.png');a.save(tmp_path/'alpha.png')
 prepare_reviewed_source(tmp_path/'src.png',tmp_path/'prepared',mask=tmp_path/'alpha.png',mask_coordinates='display')
 Image.new('RGB',(1024,1024),(190,180,170)).save(tmp_path/'plate.png')
 plane=Image.new('L',(1024,1024),0);ImageDraw.Draw(plane).rectangle((0,800,1023,1023),fill=255);plane.save(tmp_path/'plane.png')
 opts=dict(baseline=900,cast_offset=(90,30),support_plane_path=tmp_path/'plane.png')
 x=compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'default',**opts)
 y=compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'explicit',shadow_model='legacy',**opts)
 assert (tmp_path/'default/candidate.png').read_bytes()==(tmp_path/'explicit/candidate.png').read_bytes()
 z=compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'soft',shadow_model='height_soft_linear',**opts)
 assert z['native_pixel_audit']['status']=='MATCH' and z['shadow_light_geometry']=='DESIGNER_DECLARED_NOT_INFERRED'
 assert not np.array(Image.open(tmp_path/'soft/shadow-alpha.png'))[:800].any()
 assert (tmp_path/'soft/candidate.png').read_bytes()!=(tmp_path/'default/candidate.png').read_bytes()
 with pytest.raises(ValueError,match='exists'):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'soft',**opts)
 with pytest.raises(ValueError):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'invalid',shadow_softness=float('nan'),contact=False,**opts)
 assert not (tmp_path/'invalid').exists()

def test_cli_explicit_shadow_switch(monkeypatch,tmp_path):
 from renderello import cli,scene_plate
 captured=[]
 monkeypatch.setattr(scene_plate,'compose_plate',lambda *a,**k:captured.append(k) or {})
 assert cli.main(['scene-plate','--prepared','p','--background','b','--baseline','900','--out',str(tmp_path/'out'),'--shadow-model','height_soft_linear','--shadow-softness','1.5'])==0
 assert captured[0]['shadow_model']=='height_soft_linear' and captured[0]['shadow_softness']==1.5
