import hashlib
import numpy as np
import pytest
from renderello.checked_research.reviewed_subset import assemble_subset

@pytest.fixture
def data():
 s=np.arange(8*9*3,dtype=np.uint8).reshape(8,9,3)
 c=dict(name='chosen component',alpha=np.array([[0,100],[255,90]],np.uint8),crop=[2,3,4,5],source_crop_rgb_sha256=hashlib.sha256(s[3:5,2:4].tobytes()).hexdigest(),selection_note='Explicit source review, not truth.')
 return s,c

def test_exact_union(data):
 s,c=data;prior=s.copy();a=c['alpha'].copy();rgba,m=assemble_subset(s,[c],native_pairing_reviewed=True)
 assert np.array_equal(rgba[:,:,:3],s) and np.array_equal(s,prior) and np.array_equal(c['alpha'],a)
 expected=np.zeros((8,9),np.uint8);expected[3:5,2:4]=a
 assert np.array_equal(rgba[:,:,3],expected) and m['union_support_pixels']==3
 assert m['semantic_object_count'] is None and m['publication_accepted'] is False

def test_overlap_is_max_not_additive(data):
 s,c=data;other=dict(c,name='second',alpha=np.full((2,2),128,np.uint8));out,m=assemble_subset(s,[c,other],native_pairing_reviewed=True)
 assert np.array_equal(out[3:5,2:4,3],np.maximum(c['alpha'],other['alpha'])) and m['overlap_support_pixels']==3

@pytest.mark.parametrize('flag',[False,None,1,'yes'])
def test_review_required(data,flag):
 s,c=data
 with pytest.raises(ValueError):assemble_subset(s,[c],native_pairing_reviewed=flag)

@pytest.mark.parametrize('field,value',[
 ('name',''),('name',' '*2),('name',2),('name','a'*129),('selection_note',''),('selection_note',None),('selection_note','a'*2049),
 ('crop',[0,0,1]),('crop',[-1,0,1,1]),('crop',[0,0,10,1]),('crop',[2,3,2,5]),('crop',[True,3,4,5]),('crop',[2.,3,4,5]),
 ('alpha',np.zeros((2,2),np.uint8)),('alpha',np.ones((3,2),np.uint8)),('alpha',np.ones((2,2),float)),('alpha',[[0,1],[1,0]]),
 ('source_crop_rgb_sha256','wrong')])
def test_bad_component(data,field,value):
 s,c=data;c=dict(c);c[field]=value
 with pytest.raises(ValueError):assemble_subset(s,[c],native_pairing_reviewed=True)

@pytest.mark.parametrize('case',['missing','extra','duplicate','none','empty','too_many','tuple'])
def test_bad_collections(data,case):
 s,c=data;cs=[c]
 if case=='missing':c.pop('selection_note')
 if case=='extra':c['unexpected']=True
 if case=='duplicate':cs=[c,c]
 if case=='none':cs=[None]
 if case=='empty':cs=[]
 if case=='too_many':cs=[dict(c,name=str(i)) for i in range(17)]
 if case=='tuple':cs=(c,)
 with pytest.raises(ValueError):assemble_subset(s,cs,native_pairing_reviewed=True)

@pytest.mark.parametrize('source',[None,np.zeros((3,3),np.uint8),np.zeros((3,3,4),np.uint8),np.zeros((3,3,3),float),np.zeros((1,3,3),np.uint8),np.zeros((1500,1500,3),np.uint8)])
def test_bad_source(data,source):
 with pytest.raises(ValueError):assemble_subset(source,[data[1]],native_pairing_reviewed=True)
