import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_chromatic_floor import tint_reviewed_chromatic_floor
from renderello.reviewed_source_floor import tint_reviewed_source_floor


def fixture():
    rgb=np.full((20,20,3),255,np.uint8); rgb[10,8:12]=[170,130,120]
    a=np.zeros((20,20),np.uint8);a[2:9,5:15]=255;a[9,5:15]=128
    s=np.zeros_like(a);s[9:13,5:15]=255
    b=np.full_like(rgb,[235,225,215])
    return rgb,a,b,s


def run(values,**changes):
    r,a,b,s=values
    kw=dict(source_sha256=digest(r),alpha_sha256=digest(a),background_sha256=digest(b),selection_sha256=digest(s),reviewed=True,source_floor='white_opaque_with_color_reflection',review_note='Synthetic unit-test floor only')
    kw.update(changes)
    return tint_reviewed_chromatic_floor(r,a,b,s,**kw)


def test_colored_floor_preserves_native_and_outside():
    values=fixture();r,a,b,s=values;before=[v.copy() for v in values]
    out,m=run(values);base=np.rint(r*(a[:,:,None]/255)+b*(1-a[:,:,None]/255)).astype(np.uint8)
    assert np.array_equal(out[a==255],r[a==255])
    assert np.array_equal(out[s==0],base[s==0])
    assert out[10,8,0]>out[10,8,1]>out[10,8,2]
    assert m['accepted'] is False and m['recovered_emission'] is False
    assert all(np.array_equal(x,y) for x,y in zip(values,before))


@pytest.mark.parametrize('changes',[{'reviewed':False},{'source_floor':'neutral_white_opaque'},{'review_note':''},{'source_sha256':'bad'},{'alpha_sha256':'bad'},{'background_sha256':'bad'},{'selection_sha256':'bad'}])
def test_review_and_bindings(changes):
    with pytest.raises(ValueError):run(fixture(),**changes)


@pytest.mark.parametrize('mode',['dark','nonflat','wide','no_white','no_opaque','empty'])
def test_limits(mode):
    r,a,b,s=fixture()
    if mode=='dark':b[:]=20
    if mode=='nonflat':b[0,0]=200
    if mode=='wide':s[:]=255
    if mode=='no_white':r[:]=[150,100,90]
    if mode=='no_opaque':a[a==255]=254
    if mode=='empty':s[:]=0
    with pytest.raises(ValueError):run((r,a,b,s))


def test_old_neutral_tool_still_rejects_color():
    r,a,b,s=fixture()
    with pytest.raises(ValueError,match='not neutral'):
        tint_reviewed_source_floor(r,a,b,s,source_sha256=digest(r),alpha_sha256=digest(a),background_sha256=digest(b),selection_sha256=digest(s),reviewed=True,source_floor='neutral_white_opaque',review_note='Test')


@pytest.mark.parametrize('which',['rgb','alpha','background','selection'])
def test_rejects_float_arrays(which):
    values=list(fixture());index=['rgb','alpha','background','selection'].index(which)
    values[index]=values[index].astype(float)
    with pytest.raises(ValueError):run(values)


def test_fractional_selection_uses_explicit_srgb_formula():
    r,a,b,s=fixture();s[10,8]=64
    out,_=run((r,a,b,s))
    expected=np.rint(b[10,8]*((1-64/255)+(64/255)*(r[10,8].astype(float)/255))).astype(np.uint8)
    assert np.array_equal(out[10,8],expected)


def test_no_silent_resize():
    r,a,b,s=fixture()
    with pytest.raises(ValueError):run((r,a,b[:10],s))


def test_one_megapixel_cap():
    r=np.full((1001,1001,3),255,np.uint8);a=np.full((1001,1001),255,np.uint8);s=np.zeros_like(a)
    with pytest.raises(ValueError,match='1MP'):run((r,a,r.copy(),s))
