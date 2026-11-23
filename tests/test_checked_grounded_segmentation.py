import json
import numpy as np
from PIL import Image
import pytest
from renderello.checked_research import grounded_segmentation as s

@pytest.fixture
def files(tmp_path):
    src=tmp_path/'source.png'; Image.new('RGB',(10,8),(20,40,80)).save(src)
    v=dict(source_sha256=s.sha256(src),native_size=[10,8],source_resampled=False,status='PROPOSALS_NOT_SEGMENTATIONS',proposals=[dict(proposal_id=0,query='tool.',box=[1,1,9,7],model_confidence=.5,predicted_text='tool')])
    p=tmp_path/'proposals.json';p.write_text(json.dumps(v));return src,p,v

def valid_result(image,p):
    a=np.zeros((3,image.height,image.width),bool);a[:,2:6,2:8]=True
    return dict(proposal_id=p['proposal_id'],masks=a,scores=np.array([.8,.9,.7]),seconds=.1,includes_image_embedding=True)

def run(files,tmp_path,worker=valid_result):
    src,p,_=files
    return s.run(src,p,tmp_path/'out',source_sha256=s.sha256(src),proposals_sha256=s.sha256(p),segmenter=worker)

def test_full_native_export(files,tmp_path):
    m=run(files,tmp_path)
    assert json.loads((tmp_path/'out/manifest.json').read_text())==m
    rgb=np.array(Image.open(files[0]))
    for name in m['records'][0]['candidates']:
        a=np.array(Image.open(tmp_path/'out'/name));assert np.array_equal(a[:,:,:3],rgb)
        assert set(np.unique(a[:,:,3]))=={0,255}
    assert m['accuracy'] is None and m['completeness'] is None and not m['publication_accepted']
    assert m['records'][0]['selected_candidate'] is None
    assert m['binary_masks_not_matting']
    with pytest.raises(ValueError):run(files,tmp_path)

@pytest.mark.parametrize('field,value',[('source_sha256','0'*64),('native_size',[8,10]),('source_resampled',True),('status','VERIFIED'),('proposals',[]),('proposals',[{}])])
def test_invalid_manifest(files,tmp_path,field,value):
    src,p,v=files;v[field]=value;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):run(files,tmp_path)
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('field,value',[('proposal_id',True),('proposal_id',1),('box',[-1,0,5,5]),('box',[0,0,10,9]),('model_confidence',float('nan')),('predicted_text',''),('query','')])
def test_invalid_proposal(files,tmp_path,field,value):
    src,p,v=files;v['proposals'][0][field]=value;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):run(files,tmp_path)

@pytest.mark.parametrize('shape,dtype', [((3,8,10),'uint8'),((1,8,10),'bool'),((3,10,8),'bool'),((3,8,10),'float64')])
def test_mask_shape_dtype_refused(shape,dtype):
    with pytest.raises(ValueError):s.checked_masks(np.zeros(shape,dtype=dtype),[.1,.2,.3],(10,8))

@pytest.mark.parametrize('scores',[[1,2],[1,float('nan'),2],[1,float('inf'),2],['a','b','c'],[1+2j,2,3],[True,False,True]])
def test_bad_scores(scores):
    a=np.zeros((3,8,10),bool);a[:,1,1]=True
    with pytest.raises(ValueError):s.checked_masks(a,scores,(10,8))

@pytest.mark.parametrize('value',[True,False])
def test_trivial_masks_refused(value):
    with pytest.raises(ValueError):s.checked_masks(np.full((3,8,10),value),[.1,.2,.3],(10,8))

@pytest.mark.parametrize('field,value',[('proposal_id',1),('seconds',-1),('seconds',float('inf')),('seconds',True),('includes_image_embedding',None)])
def test_changed_output_refused(files,tmp_path,field,value):
    def worker(image,p):
        r=valid_result(image,p);r[field]=value;return r
    with pytest.raises(ValueError):run(files,tmp_path,worker)
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('which',['source','proposals'])
def test_hash_failure_before_inference(files,tmp_path,which):
    src,p,_=files
    def fail(*a):raise AssertionError('inference must not start')
    with pytest.raises(ValueError):s.run(src,p,tmp_path/'out',source_sha256='0'*64 if which=='source' else s.sha256(src),proposals_sha256='0'*64 if which=='proposals' else s.sha256(p),segmenter=fail)

def test_no_largest_or_best_score_selection(files,tmp_path):
    src,p,v=files;v['proposals'].append(dict(v['proposals'][0],proposal_id=1,box=[0,0,3,3]));p.write_text(json.dumps(v))
    m=run(files,tmp_path)
    assert len(m['records'])==2 and sum(len(r['candidates']) for r in m['records'])==6
    assert all(r['selected_candidate'] is None for r in m['records'])

@pytest.mark.parametrize('margin',[None,1,10,30,256])
def test_native_crop_source_rgb(files,tmp_path,margin):
    def worker(image,p):
        a=np.zeros((3,image.height,image.width),bool);a[:,1:3,1:3]=True
        return dict(proposal_id=p['proposal_id'],masks=a,scores=[.2,.3,.4],seconds=.1,includes_image_embedding=True)
    src,p,_=files
    m=s.run(src,p,tmp_path/'out',source_sha256=s.sha256(src),proposals_sha256=s.sha256(p),segmenter=worker,crop_margin=margin)
    assert m['crop_margin']==margin
    for name in m['records'][0]['candidates']:
        actual=np.array(Image.open(tmp_path/'out'/name))
        assert np.array_equal(actual[:,:,:3],np.array(Image.open(src)))

@pytest.mark.parametrize('margin',[0,-1,257,True,False,1.5,'10',float('nan')])
def test_invalid_margin_before_inference(files,tmp_path,margin):
    src,p,_=files
    def fail(*a):raise AssertionError('no inference')
    with pytest.raises(ValueError):s.run(src,p,tmp_path/'out',source_sha256=s.sha256(src),proposals_sha256=s.sha256(p),segmenter=fail,crop_margin=margin)
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('box',[[0,0,2,2],[8,6,10,8],[2.2,2.4,6.8,5.1]])
def test_crop_geometry_translation_and_pixels(box):
    rgb=np.arange(10*8*3,dtype=np.uint8).reshape(8,10,3);im=Image.fromarray(rgb)
    sub,local,crop=s.native_crop(im,dict(proposal_id=0,box=box),1)
    l,t,r,b=crop
    assert np.array_equal(np.array(sub),rgb[t:b,l:r])
    assert local['box']==[box[0]-l,box[1]-t,box[2]-l,box[3]-t]
    masks=np.zeros((3,b-t,r-l),bool);masks[:,0,0]=True
    restored,scores=s.restore_native_masks(masks,[.3,.4,.5],crop,im.size)
    assert restored.sum()==3 and restored[:,t,l].all()
    assert restored.shape==(3,8,10)

@pytest.mark.parametrize('crop',[(-1,0,3,3),(0,-1,3,3),(0,0,11,3),(0,0,3,9),(3,3,3,4)])
def test_restore_bad_bounds(crop):
    with pytest.raises(ValueError):s.restore_native_masks(np.zeros((3,3,3),bool),[.1,.2,.3],crop,(10,8))

@pytest.mark.parametrize('box',[[-1,0,2,2],[0,0,11,2],[0,0,2,9],[2,2,1,1],[0,0,float('nan'),3]])
def test_crop_bad_box(box):
    with pytest.raises(ValueError):s.native_crop(Image.new('RGB',(10,8)),dict(box=box),1)
