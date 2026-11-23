from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello.matting_alpha_prepare import prepare_alpha_only
from renderello.image_intake import IntakeError


def fixture(tmp_path):
    src=tmp_path/'src.png';rgb=np.full((60,50,3),230,np.uint8);rgb[10:50,15:35]=(20,65,140);Image.fromarray(rgb).save(src)
    bundle=tmp_path/'matting';bundle.mkdir();Image.fromarray(rgb).save(bundle/'source.png')
    a=np.zeros((60,50),np.uint8);a[10:50,15:35]=255;a[9,15:35]=128;Image.fromarray(a).save(bundle/'refined-alpha.png')
    tri=np.zeros((60,50),np.uint8);tri[9:51,14:36]=128;tri[12:48,17:33]=255;Image.fromarray(tri).save(bundle/'trimap.png')
    # Deliberately untrusted/estimated foreground is never consumed.
    Image.new('RGB',(50,60),'red').save(bundle/'foreground.png')
    m={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    (bundle/'matting-manifest.json').write_text(json.dumps(m));return src,bundle,rgb


def rehash(bundle,name):
    p=bundle/'matting-manifest.json';m=json.loads(p.read_text());m['output_sha256'][name]=hashlib.sha256((bundle/name).read_bytes()).hexdigest();p.write_text(json.dumps(m))


def test_original_rgb_only_roundtrip_and_provenance(tmp_path):
    src,bundle,rgb=fixture(tmp_path);out=tmp_path/'prepared';m=prepare_alpha_only(src,bundle,out)
    assert np.array_equal(np.array(Image.open(out/'normalized-source.png')),rgb)
    assert (out/'reviewed-alpha.png').read_bytes()==(bundle/'refined-alpha.png').read_bytes()
    assert m['estimated_alpha_provenance']['protected_core_pixels']==576
    assert m['estimated_alpha_provenance']['alpha_semantics']=='ESTIMATED_NOT_GROUND_TRUTH'
    from renderello.prepared_compose import load_prepared
    assert load_prepared(out)[0]['estimated_alpha_provenance']==m['estimated_alpha_provenance']
    before={p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):prepare_alpha_only(src,bundle,out)
    assert before=={p.name:p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize('name',['source.png','refined-alpha.png','trimap.png'])
def test_tampered_required_file_refused_without_partial(tmp_path,name):
    src,bundle,_=fixture(tmp_path);(bundle/name).write_bytes(b'changed')
    with pytest.raises(IntakeError,match='hash mismatch'):prepare_alpha_only(src,bundle,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_changed_original_refused(tmp_path):
    src,bundle,_=fixture(tmp_path);Image.new('RGB',(50,60),'black').save(src)
    with pytest.raises(IntakeError,match='source hash'):prepare_alpha_only(src,bundle,tmp_path/'out')


@pytest.mark.parametrize('defect',['size','mode','pixels','core','background','trimap','metadata'])
def test_consistent_hash_does_not_hide_bad_content(tmp_path,defect):
    src,bundle,_=fixture(tmp_path)
    if defect=='size':name='refined-alpha.png';Image.new('L',(49,60)).save(bundle/name)
    elif defect=='mode':name='refined-alpha.png';Image.new('RGB',(50,60)).save(bundle/name)
    elif defect=='pixels':name='source.png';Image.new('RGB',(50,60),'red').save(bundle/name)
    elif defect=='metadata':name='refined-alpha.png';im=Image.open(bundle/name);ex=Image.Exif();ex[274]=3;im.save(bundle/name,exif=ex)
    else:
        name='trimap.png' if defect=='trimap' else 'refined-alpha.png';im=Image.open(bundle/name);a=np.array(im)
        if defect=='core':a[20,20]=254
        elif defect=='background':a[0,0]=1
        else:a[20,20]=12
        Image.fromarray(a).save(bundle/name)
    rehash(bundle,name)
    with pytest.raises(IntakeError):prepare_alpha_only(src,bundle,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_symlink_file_refused(tmp_path):
    src,bundle,_=fixture(tmp_path);target=tmp_path/'mask.png';(bundle/'refined-alpha.png').rename(target)
    try:(bundle/'refined-alpha.png').symlink_to(target)
    except OSError:pytest.skip('Symlink privilege unavailable')
    with pytest.raises(IntakeError,match='Regular bundle file'):prepare_alpha_only(src,bundle,tmp_path/'out')


@pytest.mark.parametrize('defect',['orientation','rgba'])
def test_original_coordinate_ambiguity_refused(tmp_path,defect):
    src,bundle,_=fixture(tmp_path);im=Image.open(src)
    if defect=='orientation':ex=Image.Exif();ex[274]=3;im.save(src,exif=ex)
    else:im.convert('RGBA').save(src)
    p=bundle/'matting-manifest.json';m=json.loads(p.read_text());m['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();p.write_text(json.dumps(m))
    with pytest.raises(IntakeError,match='normalized RGB'):prepare_alpha_only(src,bundle,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_missing_or_bad_manifest_refused(tmp_path):
    src,bundle,_=fixture(tmp_path);(bundle/'matting-manifest.json').write_text('[]')
    with pytest.raises(IntakeError,match='hashes required'):prepare_alpha_only(src,bundle,tmp_path/'out')


def test_cli_and_real_compose_contract(tmp_path,capsys):
    src,bundle,_=fixture(tmp_path)
    from renderello.matting_alpha_prepare import main
    assert main(['--source',str(src),'--bundle',str(bundle),'--out',str(tmp_path/'prepared')])==0
    assert json.loads(capsys.readouterr().out)['estimated_alpha_provenance']['source_rgb_policy']=='ORIGINAL_RGB_ONLY_ESTIMATED_FOREGROUND_IGNORED'
    from renderello.scene_plate import compose_plate
    plate=tmp_path/'plate.png';Image.new('RGB',(1024,1024),(210,200,190)).save(plate)
    m=compose_plate(tmp_path/'prepared',plate,tmp_path/'composed',baseline=900,shadow_model='height_soft_linear')
    assert m['native_pixel_audit']['changed_pixels']==0


def test_source_path_change_cannot_replace_checked_bytes(tmp_path,monkeypatch):
    src,bundle,rgb=fixture(tmp_path)
    import renderello.matting_alpha_prepare as bridge
    real=bridge.inspect_image
    def changing(path,*a,**kw):
        Image.new('RGB',(50,60),'red').save(src)
        return real(path,*a,**kw)
    monkeypatch.setattr(bridge,'inspect_image',changing)
    prepare_alpha_only(src,bundle,tmp_path/'out')
    assert np.array_equal(np.array(Image.open(tmp_path/'out/normalized-source.png')),rgb)


def test_source_budget_refuses_without_resize(tmp_path):
    src,bundle,_=fixture(tmp_path)
    Image.new('RGB',(1001,1000),'white').save(src)
    Image.new('RGB',(1001,1000),'white').save(bundle/'source.png')
    Image.new('L',(1001,1000),255).save(bundle/'refined-alpha.png')
    Image.new('L',(1001,1000),128).save(bundle/'trimap.png')
    for name in ['source.png','refined-alpha.png','trimap.png']:rehash(bundle,name)
    p=bundle/'matting-manifest.json';m=json.loads(p.read_text());m['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();p.write_text(json.dumps(m))
    with pytest.raises(IntakeError,match='one million'):prepare_alpha_only(src,bundle,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_symlink_bundle_directory_refused(tmp_path):
    src,bundle,_=fixture(tmp_path);link=tmp_path/'link'
    try:link.symlink_to(bundle,target_is_directory=True)
    except OSError:pytest.skip('Symlink privilege unavailable')
    with pytest.raises(IntakeError,match='Regular matting bundle'):prepare_alpha_only(src,link,tmp_path/'out')
