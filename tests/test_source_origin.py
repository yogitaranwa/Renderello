import json,subprocess,sys
import numpy as np
from PIL import Image
import pytest
from renderello.reviewed_source import prepare_reviewed_source
from renderello.scene_plate import compose_plate


def inputs(tmp_path):
    rgb=np.full((100,80,3),(35,70,120),np.uint8)
    Image.fromarray(rgb).save(tmp_path/'source.png')
    for name,bounds in [('tight',(20,10,60,90)),('wide',(17,8,63,93))]:
        a=np.zeros((100,80),np.uint8);x0,y0,x1,y1=bounds;a[y0:y1,x0:x1]=100;a[15:85,25:55]=255
        Image.fromarray(a).save(tmp_path/(name+'.png'))
        prepare_reviewed_source(tmp_path/'source.png',tmp_path/name,mask=tmp_path/(name+'.png'),mask_coordinates='display')
    Image.new('RGB',(1024,1024),(210,200,190)).save(tmp_path/'plate.png')


def test_alpha_bounds_do_not_shift_original_source(tmp_path):
    inputs(tmp_path)
    for name in ['tight','wide']:
        m=compose_plate(tmp_path/name,tmp_path/'plate.png',tmp_path/(name+'-out'),source_origin=(300,400),contact=False)
        assert m['original_source_origin']==[300,400]
        assert m['placement_mode']=='original_source_origin'
        im=np.array(Image.open(tmp_path/(name+'-out/candidate.png')))
        assert np.all(im[415:485,325:355]==(35,70,120))
        assert m['native_pixel_audit']['status']=='MATCH'


@pytest.mark.parametrize('origin',[None,[True,0],[1.5,0],[0],[0,0,0],'0,0',{'x':0},[0,None]])
def test_missing_or_invalid_origin_refused(tmp_path,origin):
    inputs(tmp_path)
    with pytest.raises(ValueError):compose_plate(tmp_path/'tight',tmp_path/'plate.png',tmp_path/'out',source_origin=origin)
    assert not (tmp_path/'out').exists()


def test_conflicting_modes_and_outside_refused(tmp_path):
    inputs(tmp_path)
    for kw in [dict(source_origin=(0,0),baseline=850),dict(source_origin=(0,0),center_x=440),dict(source_origin=(2000,0)),dict(source_origin=(-80,0))]:
        with pytest.raises(ValueError):compose_plate(tmp_path/'tight',tmp_path/'plate.png',tmp_path/'out',**kw)
        assert not (tmp_path/'out').exists()


def test_cli_source_origin_actual(tmp_path):
    inputs(tmp_path)
    cmd=[sys.executable,'-m','renderello.cli','scene-plate','--prepared',str(tmp_path/'wide'),'--background',str(tmp_path/'plate.png'),'--out',str(tmp_path/'out'),'--source-origin','300','400','--no-contact']
    m=json.loads(subprocess.check_output(cmd,text=True));assert m['original_source_origin']==[300,400]
    failed=subprocess.run(cmd+['--baseline','850'],capture_output=True,text=True);assert failed.returncode!=0


def test_legacy_placement_still_resolves_original_origin(tmp_path):
    inputs(tmp_path);m=compose_plate(tmp_path/'tight',tmp_path/'plate.png',tmp_path/'legacy',baseline=850)
    assert m['placement_mode']=='alpha_bounds_center_baseline'
    assert m['original_source_origin']==[m['placement'][0]-m['source_box'][0],m['placement'][1]-m['source_box'][1]]
