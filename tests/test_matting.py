import numpy as np
import pytest
from PIL import Image
from renderello.matting import make_trimap,refine_arrays,refine_file,check_trimap,guided_alpha


def case():
    rgb=np.zeros((40,40,3),np.uint8);rgb[:]=[0,180,200];rgb[9:31,9:31]=[230,220,210]
    a=np.zeros((40,40),np.uint8);a[9:31,9:31]=255
    return rgb,a


def test_trimap_has_definite_seeds_and_uncertain_boundary():
    _,a=case();t=make_trimap(a,2)
    assert set(np.unique(t))=={0,128,255};assert t[20,20]==255;assert t[9,9]==128;assert t[0,0]==0

@pytest.mark.parametrize('radius',[0,13,1.5])
def test_radius_rejected(radius):
    with pytest.raises(ValueError):make_trimap(case()[1],radius)

@pytest.mark.parametrize('value',[0,255])
def test_empty_full_mask_not_confidence(value):
    with pytest.raises(ValueError):make_trimap(np.full((20,20),value,np.uint8))


def test_guided_retains_rgb_and_definite_seeds():
    rgb,a=case();t=make_trimap(a,2);out,alpha,m=refine_arrays(rgb,a,t,'guided')
    assert np.array_equal(out,rgb);assert np.all(alpha[t==255]==255);assert np.all(alpha[t==0]==0)
    assert m['changed_rgb_pixels']==0;assert m['semantic_completeness']=='NOT_VERIFIED';assert m['quality']=='NOT_EVALUATED'


def test_guided_is_finite_on_constant_rgb():
    rgb,a=case();result=guided_alpha(np.full_like(rgb,127),a);assert np.isfinite(result).all();assert result.min()>=0;assert result.max()<=1


def test_wrong_shapes_and_types():
    rgb,a=case();t=make_trimap(a)
    with pytest.raises(ValueError):refine_arrays(rgb,a.astype('float32'),t,'guided')
    with pytest.raises(ValueError):refine_arrays(rgb,a,t[:20],'guided')
    t[0,0]=64
    with pytest.raises(ValueError):check_trimap(t,a.shape)


def test_output_original_and_manifest(tmp_path):
    import json
    rgb,a=case();Image.fromarray(rgb).save(tmp_path/'s.png');Image.fromarray(a).save(tmp_path/'a.png');o=tmp_path/'out'
    m=refine_file(tmp_path/'s.png',tmp_path/'a.png',o,method='guided')
    assert m['protected_core_rgb_max_error']==0;assert m['trimap_origin'].startswith('heuristic')
    assert (o/'source.png').exists();assert 'trimap.png' in m['output_sha256']
    with pytest.raises(ValueError):refine_file(tmp_path/'s.png',tmp_path/'a.png',o,method='guided')


def test_pixel_cap():
    with pytest.raises(ValueError):refine_arrays(np.zeros((1001,1000,3),np.uint8),np.zeros((1001,1000),np.uint8),np.zeros((1001,1000),np.uint8),'guided')


def test_foreground_estimator_cannot_change_core(monkeypatch):
    import sys,types
    rgb,a=case();t=make_trimap(a)
    fake=types.SimpleNamespace(estimate_alpha_cf=lambda i,t:t,estimate_alpha_lkm=lambda i,t:t,
         estimate_foreground_ml=lambda i,a:np.zeros_like(i))
    monkeypatch.setitem(sys.modules,'pymatting',fake)
    out,alpha,m=refine_arrays(rgb,a,t,'cf',True)
    assert np.array_equal(out[t==255],rgb[t==255]);assert m['protected_core_rgb_max_error']==0
    assert m['edge_rgb_estimated'];assert not m['source_rgb_unchanged'];assert not m['entire_product_source_exact']


def test_cli_real_subprocess(tmp_path):
    import subprocess,sys,json
    rgb,a=case();Image.fromarray(rgb).save(tmp_path/'s.png');Image.fromarray(a).save(tmp_path/'a.png')
    p=subprocess.run([sys.executable,'-m','renderello.cli','matte-refine','--source',str(tmp_path/'s.png'),'--alpha',str(tmp_path/'a.png'),'--method','guided','--out',str(tmp_path/'o')],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr;assert json.loads(p.stdout)['source_rgb_unchanged']


def test_compose_bundle_preserves_provenance_and_rejects_tamper(tmp_path):
    from renderello.matting import compose_bundle
    rgb,a=case();Image.fromarray(rgb).save(tmp_path/'s.png');Image.fromarray(a).save(tmp_path/'a.png');Image.new('RGB',(100,100),'tan').save(tmp_path/'bg.png')
    refine_file(tmp_path/'s.png',tmp_path/'a.png',tmp_path/'b',method='guided')
    for field,value in [('source_pose','front'),('material','transparent'),('framing','cropped')]:
        kw=dict(source_pose='overhead',material='opaque',framing='complete');kw[field]=value
        with pytest.raises(ValueError):compose_bundle(tmp_path/'b',tmp_path/'bg.png',tmp_path/'o',**kw)
    m=compose_bundle(tmp_path/'b',tmp_path/'bg.png',tmp_path/'o',source_pose='overhead',material='opaque',framing='complete')
    assert m['quality']=='NOT_EVALUATED';assert m['original_source_sha256'];assert (tmp_path/'o/composite.png').exists()
    Image.new('RGB',(40,40),'red').save(tmp_path/'b/foreground.png')
    with pytest.raises(ValueError,match='hash mismatch'):compose_bundle(tmp_path/'b',tmp_path/'bg.png',tmp_path/'o2',source_pose='overhead',material='opaque',framing='complete')
