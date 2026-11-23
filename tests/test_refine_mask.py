import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello.refine_mask import refine


def fixture(tmp_path):
    im=np.full((120,120,3),245,np.uint8);im[25:100,30:90]=(20,55,110)
    a=np.zeros((120,120),np.uint8);a[30:95,40:80]=255
    s=np.full((120,120),128,np.uint8);s[35:85,45:75]=255;s[:15]=0;s[25:100,30:40]=192
    paths=[tmp_path/'source.png',tmp_path/'alpha.png',tmp_path/'seeds.png']
    for x,p in zip([im,a,s],paths):Image.fromarray(x).save(p)
    return paths


def test_refine_keeps_original_seeds_and_rgb(tmp_path):
    src,a,s=fixture(tmp_path);out=tmp_path/'out';meta=refine(src,a,s,out,2)
    rgba=np.array(Image.open(out/'refined-cutout.png'));assert np.array_equal(rgba[...,:3],np.array(Image.open(src)))
    seed=np.array(Image.open(s));alpha=np.array(Image.open(out/'refined-alpha.png'))
    assert (alpha[seed==255]==255).all() and (alpha[seed==0]==0).all()
    assert (out/'initial-alpha.png').read_bytes()==a.read_bytes()
    assert meta['backend']=='seeded_grabcut_assisted' and meta['status']=='REVIEW_REQUIRED'
    assert meta['alpha_sha256']==hashlib.sha256((out/'refined-alpha.png').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='exists'):refine(src,a,s,out)


@pytest.mark.parametrize('bad',['size','rgb','invalid_value','no_fg','no_bg'])
def test_refine_rejects_bad_seed_maps(tmp_path,bad):
    src,a,s=fixture(tmp_path);im=Image.open(s)
    if bad=='size':im=im.resize((60,60))
    if bad=='rgb':im=im.convert('RGB')
    if bad=='invalid_value':im.putpixel((0,0),90)
    if bad=='no_fg':im=Image.new('L',im.size,0)
    if bad=='no_bg':im=Image.new('L',im.size,255)
    im.save(s)
    with pytest.raises(ValueError):refine(src,a,s,tmp_path/'out')
    assert not (tmp_path/'out').exists()
