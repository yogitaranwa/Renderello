import numpy as np
import pytest
from renderello.shadow_geometry import rasterize_shadow_prior


def test_identity_prior_drawn_into_returned_array():
 a,d=rasterize_shadow_prior([0,0,0,0,0],[100,120,40,20,0],(256,256));assert a.dtype==np.uint8;assert a.shape==(256,256);assert a[120,100]==1;assert set(np.unique(a))=={0,1};assert d['nonzero_pixels']==861;assert d['quality']=='NOT_EVALUATED'

def test_dtype_conversion_copy_bug_reproduced_without_opencv():
 original=np.zeros((20,20),np.float32);temporary=original.astype(np.uint8);temporary[5:10,5:10]=1;assert not original.any();assert temporary.any()

def test_rotation_does_not_alias_width_and_height():
 a,d=rasterize_shadow_prior([0,0,0,0,.1],[100,120,40,20,0],(256,256));assert d['rotated_box'][2:4]==[20,40];assert a.any()

def test_offset_and_scale():
 a,d=rasterize_shadow_prior([1,1,np.log(2),np.log(.5),0],[100,120,40,20,0],(256,256));assert d['rotated_box'][:4]==[140,140,80,10];assert a[140,140]==1;assert a[120,100]==0

def test_determinism_and_no_input_mutation():
 r=np.zeros(5);f=np.array([100,120,40,20,0]);b=f.copy();a,d=rasterize_shadow_prior(r,f);a2,d2=rasterize_shadow_prior(r,f);assert np.array_equal(a,a2);assert d==d2;assert np.array_equal(f,b)

@pytest.mark.parametrize('r,f,size',[(np.zeros(4),[10,10,4,4,0],(20,20)),([0,0,float('nan'),0,0],[10,10,4,4,0],(20,20)),([0,0,0,0,0],[10,10,0,4,0],(20,20)),([0,0,9,0,0],[10,10,4,4,0],(20,20)),([0,0,0,0,0],[10,10,4,4,0],(0,20)),([0,0,0,0,0],[10,10,4,4,0],(20.1,20)),([0,0,0,0,0],[10,10,float('inf'),4,0],(20,20))])
def test_invalid_values(r,f,size):
 with pytest.raises(ValueError):rasterize_shadow_prior(r,f,size)

from renderello.shadow_geometry import add_shape_prior


def test_centroids_once_per_step_and_original_unchanged():
 b=np.zeros((1,2,4),np.float32);labels=np.array([[7,9]]);centroids={7:np.ones(4),9:np.ones(4)*2}
 for _ in range(50):a=add_shape_prior(b,labels,centroids)
 assert np.array_equal(a[0,0],np.ones(4));assert np.array_equal(a[0,1],np.ones(4)*2);assert not b.any()


def test_missing_centroid_keeps_base():
 b=np.full((1,1,3),.5,np.float32);a=add_shape_prior(b,np.array([[4]]),{});assert np.array_equal(a,b);assert not np.shares_memory(a,b)

@pytest.mark.parametrize('b,labels,c',[(np.zeros((1,1,3),np.int32),np.array([[1]]),{}),(np.zeros((1,1,3)),np.array([[1.5]]),{}),(np.zeros((1,1,3)),np.array([[1]]),{1:[0,0]}),(np.zeros((1,1,3)),np.array([[1]]),{1:[0,np.nan,0]}),(np.zeros((1,1,3)),np.array([[1,2]]),{}),(np.full((1,1,3),np.inf),np.array([[1]]),{})])
def test_bad_embedding_values(b,labels,c):
 with pytest.raises(ValueError):add_shape_prior(b,labels,c)


def test_embedding_overflow_rejected():
 b=np.full((1,1,3),65000,dtype=np.float16)
 with pytest.raises(ValueError):add_shape_prior(b,np.array([[1]]),{1:[65000]*3})
