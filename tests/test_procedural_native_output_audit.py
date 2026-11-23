"""Saved-output procedural audit. No external provider/model calls."""
import json
import numpy as np
from PIL import Image
from renderello.schemas import CutoutResult, Aspect
from renderello.templates import get_template
from renderello.compose import composite, audit_composite


def fixture(tmp_path, size):
    w,h=size
    rgb=np.zeros((h,w,3),np.uint8)
    rgb[:]=[80,110,140]
    alpha=np.full((h,w),255,np.uint8)
    rp=tmp_path/'rgba.png';ap=tmp_path/'alpha.png'
    Image.fromarray(np.dstack([rgb,alpha])).save(rp)
    Image.fromarray(alpha).save(ap)
    return rgb,CutoutResult(rp,ap,(0,0,w,h),'synthetic',1.)


def test_native_saved_output_matches(tmp_path):
    source,co=fixture(tmp_path,(100,200));t=get_template('clean_studio')
    out=composite(co,t,Aspect.SQUARE,0);p=tmp_path/'saved.png';out.save(p)
    a=audit_composite(co,source,np.array(Image.open(p)),Aspect.SQUARE,0,t.key)
    assert a['status']=='MATCH' and a['changed_pixels']==0
    assert a['fully_opaque_source_pixels']==20000
    assert a['quality_acceptance']=='NOT_EVALUATED'


def test_saved_tampered_product_is_mismatch(tmp_path):
    source,co=fixture(tmp_path,(100,200));t=get_template('clean_studio')
    out=np.array(composite(co,t,Aspect.SQUARE,0))
    a=audit_composite(co,source,out,Aspect.SQUARE,0,t.key)
    x,y,w,h=a['output_box'];out[y,x,0]^=1
    b=audit_composite(co,source,out,Aspect.SQUARE,0,t.key)
    assert b['status']=='MISMATCH' and b['changed_pixels']==1


def test_resized_product_never_claims_native_match(tmp_path):
    source,co=fixture(tmp_path,(1000,1500));t=get_template('clean_studio')
    out=np.array(composite(co,t,Aspect.SQUARE,0))
    a=audit_composite(co,source,out,Aspect.SQUARE,0,t.key)
    assert a['status']=='NATIVE_SIZE_CHANGED'
    assert a['native_rgb_exact'] is None and a['changed_pixels'] is None


def test_provider_does_not_promise_exactness():
    from renderello.scene_providers import ProceduralProvider
    assert 'stay source-exact' not in ProceduralProvider.quality_note
    assert 'resized' in ProceduralProvider.quality_note
