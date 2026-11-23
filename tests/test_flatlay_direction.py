from dataclasses import replace
import numpy as np
import pytest
from renderello.contact_spans import ContactPolicy,render_contact
from renderello.candidate_diagnostics import digest


def fixture():
    rgb=np.full((80,90,3),240,np.uint8);a=np.zeros((80,90),np.uint8);a[20:60,25:65]=255;rgb[a==255]=[25,70,100]
    p=ContactPolicy('flatlay','opaque','complete','overhead',digest(rgb),digest(a),reviewed=True,length=12,decay=2)
    return rgb,a,np.full_like(rgb,225),p


def test_explicit_direction_and_source_invariants():
    r,a,b,p=fixture();out,shade,m=render_contact(r,a,b,[],replace(p,flatlay_cast_offset=(4,3)))
    assert shade[40,66]>shade[40,23]
    assert shade[61,40]>shade[18,40]
    assert not shade[a>=128].any() and np.array_equal(out[a==255],r[a==255])
    assert m['method']=='authored_flatlay_offset_silhouette' and not m['light_inferred']
    assert not m['source_height_inferred'] and m['flatlay_cast_offset']==[4,3]


def test_default_identical_to_explicit_zero():
    r,a,b,p=fixture();x,s,m=render_contact(r,a,b,[],p);y,t,_=render_contact(r,a,b,[],replace(p,flatlay_cast_offset=(0,0)))
    # OpenCV precise distance fields can differ by a few float32 ULPs between
    # identical threaded evaluations. Stored RGB must still be exactly equal.
    assert np.array_equal(x,y) and np.allclose(s,t,rtol=0,atol=1e-8)
    assert s[40,66]==s[40,23] and m['method']=='flatlay_finite_outside_distance'


@pytest.mark.parametrize('offset',[None,(1,),[True,0],(1.,0),(17,0),(0,-17),'1,2'])
def test_bounded_explicit_offsets(offset):
    with pytest.raises(ValueError):replace(fixture()[-1],flatlay_cast_offset=offset)


def test_no_upright_bypass():
    with pytest.raises(ValueError):replace(fixture()[-1],camera='front',source_pose='front',flatlay_cast_offset=(1,1))


def test_no_canvas_clip():
    r,a,b,p=fixture();a[20:60,75:89]=255;p=replace(p,alpha_sha256=digest(a),flatlay_cast_offset=(10,0))
    with pytest.raises(ValueError,match='beyond canvas'):render_contact(r,a,b,[],p)


def test_opacity_zero_is_identity():
    r,a,b,p=fixture();out,s,_=render_contact(r,a,b,[],replace(p,opacity=0,flatlay_cast_offset=(-4,-3)))
    assert not s.any() and np.array_equal(out[a==0],b[a==0])
