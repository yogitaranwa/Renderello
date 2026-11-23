import numpy as np
import pytest
from renderello.checked_research.intrinsic_gain import intrinsic_gain

@pytest.fixture
def data():
 rng=np.random.default_rng(4);s=rng.integers(0,256,(17,19,3),dtype=np.uint8);alpha=np.zeros((17,19),np.uint8);alpha[3:14,4:15]=255;alpha[2,4:15]=128;pred=rng.uniform(.1,.9,(3,17,19,3)).astype('float32');return s,alpha,pred

def test_identity(data):
 s,a,p=data;out,g,m=intrinsic_gain(s,a,p,alignment_reviewed=True,shading_strength=0,residual_suppression=0);assert np.array_equal(out[:,:,:3],s) and np.array_equal(out[:,:,3],a) and np.all(g==1) and m['source_rgb_exact']

def test_invariants(data):
 s,a,p=data;copies=[x.copy() for x in data];out,g,m=intrinsic_gain(s,a,p,alignment_reviewed=True);assert np.array_equal(out[:,:,3],a) and np.array_equal(out[:,:,:3][a==0],s[a==0]);assert np.isfinite(g).all() and g.min()>=.65 and g.max()<=1.45;assert m['preencode_out_of_gamut_pixels']==0 and m['physical_relighting'] is False and m['independent_accuracy'] is None and m['estimated_rgb_replaces_source'] is False
 for x,y in zip(data,copies):assert np.array_equal(x,y)
 assert np.allclose(np.repeat(g[:,:,None],3,axis=2),np.dstack([g,g,g]))

@pytest.mark.parametrize('flag',[False,None,1,'yes'])
def test_review(data,flag):
 with pytest.raises(ValueError):intrinsic_gain(*data,alignment_reviewed=flag)

@pytest.mark.parametrize('key,value', [('sigma',0),('sigma',-1),('sigma',101),('sigma',float('nan')),('sigma',True),('shading_strength',-1),('shading_strength',1.01),('shading_strength',True),('residual_suppression',-1),('residual_suppression',2),('residual_suppression',float('inf')),('gain_bounds',None),('gain_bounds',(.1,.9)),('gain_bounds',(1.1,1.5)),('gain_bounds',(0,2)),('gain_bounds',(.5,2.1)),('gain_bounds',(True,1.5)),('gain_bounds',(float('nan'),1.4)),('gain_bounds',(1,1,1))])
def test_controls(data,key,value):
 with pytest.raises(ValueError):intrinsic_gain(*data,alignment_reviewed=True,**{key:value})

@pytest.mark.parametrize('index,kind',[(0,'none'),(0,'dtype'),(0,'dimensions'),(1,'none'),(1,'dtype'),(1,'dimensions'),(1,'empty'),(1,'noopaque'),(2,'none'),(2,'dtype'),(2,'dimensions'),(2,'nan'),(2,'negative'),(2,'over')])
def test_bad_arrays(data,index,kind):
 d=list(data)
 if kind=='none':d[index]=None
 if kind=='dtype':d[index]=d[index].astype('float32' if index!=2 else 'uint8')
 if kind=='dimensions':d[index]=d[index][0]
 if kind in ['empty','noopaque']:d[index]=np.zeros_like(d[index])+(128 if kind=='noopaque' else 0)
 if kind in ['nan','negative','over']:d[index]=d[index].copy();d[index].flat[0]={'nan':float('nan'),'negative':-.01,'over':1.01}[kind]
 with pytest.raises(ValueError):intrinsic_gain(*d,alignment_reviewed=True)

def test_unusable_maps(data):
 s,a,p=data;p=p.copy();p[1]=0
 with pytest.raises(ValueError):intrinsic_gain(s,a,p,alignment_reviewed=True)
 out,g,m=intrinsic_gain(s,a,p,alignment_reviewed=True,shading_strength=0,residual_suppression=0);assert np.array_equal(out[:,:,:3],s)

def test_clipping_cap(data):
 s,a,p=data;s[:]=255;out,g,m=intrinsic_gain(s,a,p,alignment_reviewed=True,residual_suppression=0);assert g.max()<=1 and m['preencode_out_of_gamut_pixels']==0
