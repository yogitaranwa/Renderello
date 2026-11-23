import numpy as np
import pytest
from renderello.reviewed_source_floor import tint_reviewed_source_floor
from renderello.candidate_diagnostics import digest


def case():
    rgb=np.full((20,24,3),255,np.uint8);rgb[6:12,8:16]=40;rgb[12:15,8:16]=180
    a=np.zeros((20,24),np.uint8);a[6:12,8:16]=255;a[11,8:16]=128
    bg=np.full_like(rgb,[220,210,200]);m=np.zeros_like(a);m[10:16,5:20]=255
    return rgb,a,bg,m


def run(arrays,**kw):
    rgb,a,bg,m=arrays
    opts=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),background_sha256=digest(bg),selection_sha256=digest(m),reviewed=True,source_floor='neutral_white_opaque',review_note='Reviewed synthetic white floor, not physical truth')
    opts.update(kw)
    return tint_reviewed_source_floor(rgb,a,bg,m,**opts)


def test_identity_and_source_untouched():
    arrays=case();copies=[a.copy()for a in arrays];out,meta=run(arrays);rgb,a,bg,m=arrays
    assert all(np.array_equal(x,y)for x,y in zip(arrays,copies))
    assert np.array_equal(out[a==255],rgb[a==255])
    assert np.array_equal(out[0],bg[0])
    assert np.all(out[13,10]<bg[13,10]);assert meta['accepted'] is False
    assert meta['recovered_shadow'] is False and meta['default_pipeline_changed'] is False


def test_white_pixels_do_not_invent_shadow():
    rgb,a,bg,m=case();rgb[:]=255;out,_=run((rgb,a,bg,m));baseline=np.rint(rgb*(a[:,:,None]/255)+bg*(1-a[:,:,None]/255)).astype('uint8');assert np.array_equal(out,baseline)


def test_empty_selection_refused():
    rgb,a,bg,m=case();m[:]=0
    with pytest.raises(ValueError):run((rgb,a,bg,m))


@pytest.mark.parametrize('key,value',[('reviewed',False),('source_floor','gray'),('review_note',''),('source_sha256','0'*64),('alpha_sha256','0'*64),('background_sha256','0'*64),('selection_sha256','0'*64)])
def test_review_and_binding_refusals(key,value):
    with pytest.raises(ValueError):run(case(),**{key:value})


@pytest.mark.parametrize('kind',['dark','nonflat','chromatic','noopaque','badmask','mismatch'])
def test_unsupported_refused(kind):
    rgb,a,bg,m=case()
    if kind=='dark':bg[:]=20
    if kind=='nonflat':bg[0,0]=200
    if kind=='chromatic':rgb[13,10]=[255,200,180]
    if kind=='noopaque':a[a==255]=254
    if kind=='badmask':m=m.astype(float)
    if kind=='mismatch':bg=bg[:-1]
    with pytest.raises(ValueError):run((rgb,a,bg,m))
