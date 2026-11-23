from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello.cached_native_scene_cli import run, main


def inputs(tmp_path):
    source=tmp_path/'s.png';alpha=tmp_path/'a.png';scene=tmp_path/'b.png'
    Image.new('RGB',(30,40),(180,50,20)).save(source)
    a=np.zeros((40,30),np.uint8);a[5:35,5:25]=255;a[4]=128
    Image.fromarray(a).save(alpha);Image.new('RGB',(100,100),(210,210,210)).save(scene)
    return source,alpha,scene


def test_offline_cli_preserves_snapshots_and_actual_visible_opaque_pixels(tmp_path):
    s,a,b=inputs(tmp_path);out=tmp_path/'result'
    assert main(['--source',str(s),'--alpha',str(a),'--scene',str(b),'--out',str(out),'--x','20','--y','10'])==0
    m=json.loads((out/'manifest.json').read_text());assert not (out/'INCOMPLETE').exists()
    assert m['provider_calls']==m['credits_spent']==0 and not m['photo_accepted']
    for name,path in [('source',s),('alpha',a),('scene',b)]:
        assert (out/(name+'.png')).read_bytes()==path.read_bytes()
        assert m['input_file_sha256'][name]==hashlib.sha256(path.read_bytes()).hexdigest()
    result=np.array(Image.open(out/'composite.png'));assert np.all(result[15:45,25:45]==[180,50,20])


def test_output_preserved_and_out_of_bounds_creates_no_directory(tmp_path):
    s,a,b=inputs(tmp_path);out=tmp_path/'result';out.mkdir();(out/'keep').write_text('prior')
    with pytest.raises(FileExistsError):run(s,a,b,out,0,0)
    assert (out/'keep').read_text()=='prior'
    other=tmp_path/'invalid'
    with pytest.raises(ValueError):run(s,a,b,other,90,90)
    assert not other.exists()


@pytest.mark.parametrize('kind',['RGB-alpha','L-source','16bit-alpha','profile-source','oriented-source','color-key-source','bad-size'])
def test_no_implicit_mode_depth_profile_or_dimension_conversion(tmp_path,kind):
    s,a,b=inputs(tmp_path)
    if kind=='RGB-alpha':Image.new('RGB',(30,40)).save(a)
    elif kind=='L-source':Image.new('L',(30,40)).save(s)
    elif kind=='16bit-alpha':Image.fromarray(np.zeros((40,30),np.uint16)).save(a)
    elif kind=='profile-source':Image.new('RGB',(30,40)).save(s,icc_profile=b'unknown')
    elif kind=='oriented-source':
        exif=Image.Exif();exif[274]=6;Image.new('RGB',(30,40)).save(s,exif=exif)
    elif kind=='color-key-source':Image.new('RGB',(30,40)).save(s,transparency=(0,0,0))
    else:Image.new('L',(40,30)).save(a)
    out=tmp_path/'refused'
    with pytest.raises(ValueError):run(s,a,b,out,0,0)
    assert not out.exists()


def test_symlink_input_is_not_followed(tmp_path):
    s,a,b=inputs(tmp_path);link=tmp_path/'link.png';link.symlink_to(s)
    with pytest.raises(OSError):run(link,a,b,tmp_path/'result',0,0)


def test_fifo_is_refused_without_blocking(tmp_path):
    import os
    if not hasattr(os, 'mkfifo'):pytest.skip('POSIX FIFO fixture is unavailable on Windows; not exercised')
    s,a,b=inputs(tmp_path);fifo=tmp_path/'pipe';os.mkfifo(fifo)
    with pytest.raises(ValueError):run(fifo,a,b,tmp_path/'result',0,0)
