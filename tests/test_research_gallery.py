import json,hashlib
from pathlib import Path
from PIL import Image
import pytest
from renderello.research_gallery import build_research_gallery,DIMENSIONS


def fixture(tmp_path):
    for name,color in [('scene',(40,60,80)),('source',(200,210,220)),('product',(100,120,130))]:Image.new('RGB',(16,16),color).save(tmp_path/(name+'.png'))
    def asset(name):return dict(path=name+'.png',sha256=hashlib.sha256((tmp_path/(name+'.png')).read_bytes()).hexdigest())
    base=dict(id='scene1',kind='scene_only',attempt='first_try',intent='Texture <script>',image=asset('scene'),source=None,source_id=None,note='Rejected & retained',caption='Generated background',scores=None,reviewer='assistant_scene_inspection',critical_errors_status='Not applicable: no product')
    product={**base,'id':'product1','kind':'product_composite','image':asset('product'),'source':asset('source'),'source_id':'one','scores':dict.fromkeys(DIMENSIONS,2),'reviewer':'assistant_not_blind_or_independent','critical_errors_status':'Not independently verified'}
    m=dict(schema='renderello-research-gallery.v1',title='Test',note='Private research only',cases=[base,product]);p=tmp_path/'manifest.json';p.write_text(json.dumps(m));return p,m


def write(p,m):p.write_text(json.dumps(m))


def test_counts_and_escaping(tmp_path):
    p,m=fixture(tmp_path);m['cases'].append({**m['cases'][1],'id':'repeat','attempt':'retry'});write(p,m);r=build_research_gallery(p,tmp_path/'index.html');text=(tmp_path/'index.html').read_text();assert r['scene_attempts']==1 and r['product_composite_attempts']==2 and r['product_distinct_encoded_artifacts']==1 and r['benchmark_accepts']==0 and r['distinct_source_encoded_artifacts']==1;assert '<script>'not in text and '&lt;script&gt;'in text;assert 'NOT ACCEPTED'in text


@pytest.mark.parametrize('change', ['hash','traverse','symlink','scores','duplicate','scene_score','reviewer','missing','boolean','unknown'])
def test_refuses_bad_evidence_before_write(tmp_path,change):
    p,m=fixture(tmp_path);c=m['cases'][1]
    if change=='hash':c['image']['sha256']='0'*64
    if change=='traverse':c['image']['path']='../product.png'
    if change=='symlink':
        (tmp_path/'link').symlink_to(tmp_path.parent,target_is_directory=True);c['image']['path']='link/x.png'
    if change=='scores':c['scores'].pop(DIMENSIONS[0])
    if change=='duplicate':c['id']='scene1'
    if change=='scene_score':m['cases'][0]['scores']=c['scores']
    if change=='reviewer':c['reviewer']='independent'
    if change=='missing':c['image']['path']='missing.png'
    if change=='boolean':c['scores'][DIMENSIONS[0]]=True
    if change=='unknown':c['accepted']=True
    write(p,m)
    with pytest.raises((ValueError,FileNotFoundError)):build_research_gallery(p,tmp_path/'index.html')
    assert not(tmp_path/'index.html').exists()


def test_no_overwrite_and_portable_root(tmp_path):
    p,m=fixture(tmp_path);build_research_gallery(p,tmp_path/'index.html')
    with pytest.raises(FileExistsError):build_research_gallery(p,tmp_path/'index.html')
    with pytest.raises(ValueError):build_research_gallery(p,tmp_path/'child/index.html')
