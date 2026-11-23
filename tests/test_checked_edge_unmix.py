import numpy as np
import pytest
from renderello.checked_research.edge_unmix import unmix_edge

@pytest.fixture
def data():
 s=np.full((13,13,3),100,np.uint8);a=np.zeros((13,13),np.uint8);a[3:10,3:10]=128;a[4:9,4:9]=255;return s,a

@pytest.mark.parametrize('space',['linear','srgb'])
def test_invariants(data,space):
 s,a=data;old=s.copy();aa=a.copy();out,m=unmix_edge(s,a,pairing_reviewed=True,mixing_space=space);assert np.array_equal(out[a==255],s[a==255]) and np.array_equal(out[a==0],s[a==0]) and np.array_equal(s,old) and np.array_equal(a,aa);assert m['alpha_changed'] is False and m['boundary_accuracy'] is None and m['publication_accepted'] is False

@pytest.mark.parametrize('flag',[False,None,1,'yes'])
def test_review(data,flag):
 with pytest.raises(ValueError):unmix_edge(*data,pairing_reviewed=flag,mixing_space='srgb')

@pytest.mark.parametrize('key,value',[('mixing_space',None),('mixing_space','auto'),('regularization',0),('regularization',1.1),('regularization',float('nan')),('regularization',True),('max_distance',0),('max_distance',101),('max_distance',float('inf')),('max_distance',True)])
def test_controls(data,key,value):
 kw=dict(pairing_reviewed=True,mixing_space='srgb');kw[key]=value
 with pytest.raises(ValueError):unmix_edge(*data,**kw)

@pytest.mark.parametrize('case',['rgbnone','rgbfloat','rgbshape','anone','afloat','ashape','nofg','nobg','noedge'])
def test_bad_arrays(data,case):
 s,a=data
 if case=='rgbnone':s=None
 if case=='rgbfloat':s=s.astype(float)
 if case=='rgbshape':s=s[:,:,0]
 if case=='anone':a=None
 if case=='afloat':a=a.astype(float)
 if case=='ashape':a=a[0]
 if case=='nofg':a[a==255]=128
 if case=='nobg':a[a==0]=128
 if case=='noedge':a[a==128]=255
 with pytest.raises(ValueError):unmix_edge(s,a,pairing_reviewed=True,mixing_space='linear')

def test_known_constant_scene():
 a=np.zeros((21,21),np.uint8);a[4:17,4:17]=np.tile(np.linspace(32,224,13,dtype=np.uint8),(13,1));a[7:14,7:14]=255;f=np.array([210,30,80]);b=np.array([30,180,40]);rgb=np.round(a[:,:,None]/255*f+(1-a[:,:,None]/255)*b).astype('uint8');out,m=unmix_edge(rgb,a,pairing_reviewed=True,mixing_space='srgb');e=(a>0)&(a<255);assert np.abs(out.astype(float)[e]-f).mean()<np.abs(rgb.astype(float)[e]-f).mean()

def test_distance_refusal(data):
 s,a=data;out,m=unmix_edge(s,a,pairing_reviewed=True,mixing_space='srgb',max_distance=.1);assert np.array_equal(out,s) and m['eligible_pixels']==0
