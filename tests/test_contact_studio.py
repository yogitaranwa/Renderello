import json
import numpy as np
from PIL import Image
import pytest
from renderello.contact_studio import shadow_layers,replay_contact
from renderello.native_studio import compose_native
from test_native_studio import bundle

def a():
 v=np.zeros((120,100),np.uint8);v[10:110,20:80]=255;return v

def test_shadow_floor_contact_and_determinism():
 c,s,b,d=shadow_layers(a(),(300,300),[80,70]);assert d['floor_row']==179;assert not np.any(b[:179]);assert c[179,130]>0;assert s[185:].max()>0;assert np.array_equal(b,shadow_layers(a(),(300,300),[80,70])[2]);assert b.max()<=255

def test_gaps_not_ellipse():
 v=a();v[:]=0;v[10:110,10:20]=255;v[10:110,80:90]=255
 c,_,_,_=shadow_layers(v,(300,300),[80,70]);assert c[179,95]>0;assert c[179,130]==0

def test_direction_changes_cast_not_contact():
 p=shadow_layers(a(),(300,300),[80,70],direction=.9);n=shadow_layers(a(),(300,300),[80,70],direction=-.9);assert np.array_equal(p[0],n[0]);assert not np.array_equal(p[1],n[1]);assert np.where(p[1]>0)[1].mean()>np.where(n[1]>0)[1].mean()

@pytest.mark.parametrize('kwargs',[dict(direction=2),dict(direction=float('nan')),dict(depth=.0),dict(depth=.4),dict(cast_opacity=.9),dict(contact_opacity=-.1),dict(direction=True)])
def test_invalid_parameters(kwargs):
 with pytest.raises(ValueError):shadow_layers(a(),(300,300),[80,70],**kwargs)

@pytest.mark.parametrize('what',['dtype','empty','canvas','place','outside','floor'])
def test_invalid_geometry(what):
 v=a();size=(300,300);place=[80,70]
 if what=='dtype':v=v.astype(float)
 if what=='empty':v[:]=127
 if what=='canvas':size=(63,300)
 if what=='place':place=[1.2,2]
 if what=='outside':place=[250,70]
 if what=='floor':place=[80,190]
 with pytest.raises(ValueError):shadow_layers(v,size,place)

def test_replay_source_exact_and_prior_preserved(tmp_path):
 b=bundle(tmp_path);old=tmp_path/'old';compose_native(b,'porcelain_sweep',old,source_pose='front',material='opaque',framing='complete');before=(old/'candidate.png').read_bytes();out=tmp_path/'new';d=replay_contact(old,out,flat_floor_confirmed=True);assert d['opaque_core_rgb_max_error']==0;assert d['quality']=='NOT_EVALUATED';assert before==(old/'candidate.png').read_bytes()
 with pytest.raises(ValueError,match='exists'):replay_contact(old,out,flat_floor_confirmed=True)

def test_unapproved_floor_and_platform_rejected(tmp_path):
 b=bundle(tmp_path);old=tmp_path/'old';compose_native(b,'clay_plinth',old,source_pose='front',material='opaque',framing='complete')
 with pytest.raises(ValueError,match='declaration'):replay_contact(old,tmp_path/'new')
 with pytest.raises(ValueError,match='platform'):replay_contact(old,tmp_path/'new',flat_floor_confirmed=True)
 assert not (tmp_path/'new').exists()

def test_hash_tamper_rejected(tmp_path):
 b=bundle(tmp_path);old=tmp_path/'old';compose_native(b,'porcelain_sweep',old,source_pose='front',material='opaque',framing='complete');(old/'background.png').write_bytes(b'changed')
 with pytest.raises(ValueError,match='hash'):replay_contact(old,tmp_path/'new',flat_floor_confirmed=True)

def test_no_opaque_core_rejected(tmp_path):
 from renderello.intent_mask import sha
 b=bundle(tmp_path);old=tmp_path/'old';compose_native(b,'porcelain_sweep',old,source_pose='front',material='opaque',framing='complete');f=old/'native-product-crop.png';v=np.array(Image.open(f));v[:,:,3]=254;Image.fromarray(v).save(f);p=old/'native-studio-manifest.json';d=json.loads(p.read_text());d['output_sha256'][f.name]=sha(f);p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='opaque core'):replay_contact(old,tmp_path/'new',flat_floor_confirmed=True)
