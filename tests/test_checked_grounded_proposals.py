import json
import numpy as np
import pytest
from PIL import Image
from renderello.checked_research import grounded_proposals as g


def row(box=None,score=.5,label='tool',query='tool.'):
    return dict(query=query,boxes=[box or [1,2,8,9]],scores=[score],labels=[label])

@pytest.mark.parametrize('queries',[None,[],['A.'],['a'],[' a.'],['a. '],['a.']*2,['a'*257+'.'],[{}],['.']*13])
def test_bad_queries(queries):
    with pytest.raises(ValueError):g.validate_queries(queries)

def test_queries_retained():
    q=['barber tools.','bottle.'];assert g.validate_queries(q)==q

@pytest.mark.parametrize('box',[[-1,0,5,5],[0,0,11,5],[0,0,5,11],[5,0,5,5],[6,0,5,5],[0,0,float('nan'),5],[0,0,float('inf'),5],[0,0,5]])
def test_bad_boxes(box):
    with pytest.raises(ValueError):g.checked_proposals([row(box)],10,10)

@pytest.mark.parametrize('score',[-.1,1.1,float('nan'),float('inf')])
def test_bad_score(score):
    with pytest.raises(ValueError):g.checked_proposals([row(score=score)],10,10)

def test_count_mismatch():
    x=row();x['labels']=[]
    with pytest.raises(ValueError):g.checked_proposals([x],10,10)

def test_empty_is_not_complete():
    assert g.checked_proposals([dict(query='x.',boxes=[],scores=[],labels=[])],10,10)==[]

def test_alternative_labels_preserved():
    p=g.checked_proposals([row(label='comb',query='comb.'),row(label='razor',query='razor.')],10,10)
    groups=g.group_overlaps(p)
    assert len(groups)==1 and len(groups[0]['members'])==2
    assert [m['predicted_text'] for m in groups[0]['members']]==['comb','razor']

def test_separate_boxes_not_largest_only():
    p=g.checked_proposals([row([0,0,2,2]),row([3,3,10,10])],10,10)
    assert len(g.group_overlaps(p))==2

@pytest.mark.parametrize('threshold',[0,1.1,float('nan'),True])
def test_invalid_group_threshold(threshold):
    with pytest.raises(ValueError):g.group_overlaps([],threshold)

@pytest.fixture
def source(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(10,10),'red').save(p);return p

def test_run_roundtrip_and_exclusive(source,tmp_path):
    dest=tmp_path/'out'
    result=g.run(source,dest,source_sha256=g.sha256(source),queries=['tool.'],predictor=lambda im,q:row(query=q))
    assert json.loads((dest/'manifest.json').read_text())==result
    assert np.array_equal(np.array(Image.open(source)),np.array(Image.open(dest/'source-native.png')))
    assert result['accuracy'] is None and result['completeness'] is None
    assert not result['publication_accepted'] and result['status']=='PROPOSALS_NOT_SEGMENTATIONS'
    with pytest.raises(ValueError):g.run(source,dest,source_sha256=g.sha256(source),queries=['tool.'],predictor=lambda im,q:row())

def test_bad_hash_no_predictor(source,tmp_path):
    def fail(*a):raise AssertionError('inference must not start')
    with pytest.raises(ValueError):g.run(source,tmp_path/'out',source_sha256='0'*64,queries=['tool.'],predictor=fail)

def test_no_mode_conversion(tmp_path):
    p=tmp_path/'gray.png';Image.new('L',(10,10)).save(p)
    with pytest.raises(ValueError):g.run(p,tmp_path/'out',source_sha256=g.sha256(p),queries=['tool.'],predictor=lambda *a:row())

def test_changed_query_refused(source,tmp_path):
    with pytest.raises(ValueError):g.run(source,tmp_path/'out',source_sha256=g.sha256(source),queries=['tool.'],predictor=lambda *a:row(query='different.'))
    assert not (tmp_path/'out').exists()
