import numpy as np
import pytest
from renderello.reviewed_matting_patch import apply_reviewed_matting_patch
from renderello.candidate_diagnostics import digest


def sample():
 r=np.full((24,30,3),100,np.uint8);a=np.zeros((24,30),np.uint8);a[4:20,7:23]=255;a[6:18,6]=128;e=r.copy();e[6:18,6]=220;aa=a.copy();aa[6:18,6]=160;tri=np.full(a.shape,128,np.uint8);tri[:3]=0;tri[8:16,12:20]=255;region=np.zeros_like(a,bool);region[4:20,4:10]=True;p=np.zeros_like(region);p[8:16,12:20]=True
 arr=[r,a,e,aa,tri,region,p];keys=['source_sha256','alpha_sha256','estimated_rgb_sha256','estimated_alpha_sha256','trimap_sha256','region_sha256','protected_sha256'];k=dict(zip(keys,map(digest,arr)));k.update(reviewed=True,material='opaque_fibres',review_note='synthetic protected patch',color_mode='bounded-estimate',support_policy='reviewed-change',feather=1,max_rgb_delta=24)
 return arr,k


def test_patch_exact_invariants():
 arr,k=sample();r,a,e,aa,t,reg,p=arr;o,b,m=apply_reviewed_matting_patch(*arr,**k);assert np.all(o[6:18,6]==124);assert np.all(b[6:18,6]==160);assert np.array_equal(o[~reg],r[~reg])and np.array_equal(b[~reg],a[~reg]);assert not m['accepted']


def test_source_colour_mode():
 arr,k=sample();k['color_mode']='source';o,b,m=apply_reviewed_matting_patch(*arr,**k);assert np.array_equal(o,arr[0])


def test_opaque_colour_retained():
 arr,k=sample();arr[2][arr[1]==255]=255;k['estimated_rgb_sha256']=digest(arr[2]);o,b,m=apply_reviewed_matting_patch(*arr,**k);assert np.array_equal(o[arr[1]==255],arr[0][arr[1]==255])


@pytest.mark.parametrize('policy',['retain','reviewed-change'])
def test_support_policy(policy):
 arr,k=sample();arr[3][7,5]=200;arr[3][7,6]=0;k['estimated_alpha_sha256']=digest(arr[3]);k['support_policy']=policy;o,b,m=apply_reviewed_matting_patch(*arr,**k)
 if policy=='retain':assert b[7,5]==0 and b[7,6]==1 and m['support_removed_pixels']==0
 else:assert b[7,5]==200 and b[7,6]==0 and m['support_added_pixels']==1 and m['support_removed_pixels']==1


@pytest.mark.parametrize('key,value',[('reviewed',False),('reviewed',1),('material','glass'),('material',None),('review_note',''),('color_mode',None),('color_mode','all'),('support_policy',None),('feather',True),('feather',0),('feather',17),('max_rgb_delta',65),('max_alpha_delta',256),('max_alpha_delta',1),('source_sha256','bad'),('alpha_sha256','bad'),('estimated_rgb_sha256','bad'),('estimated_alpha_sha256','bad'),('trimap_sha256','bad'),('region_sha256','bad'),('protected_sha256','bad')])
def test_bad_controls(key,value):
 arr,k=sample();k[key]=value
 with pytest.raises(ValueError):apply_reviewed_matting_patch(*arr,**k)


@pytest.mark.parametrize('case',['shape','empty_region','region_float','empty_protected','overlap','trimap_values','no_unknown','no_protected_fg'])
def test_invalid_masks(case):
 arr,k=sample()
 if case=='shape':arr[2]=arr[2][:-1]
 if case=='empty_region':arr[5][:]=False
 if case=='region_float':arr[5]=arr[5].astype(float)
 if case=='empty_protected':arr[6][:]=False
 if case=='overlap':arr[6]|=arr[5]
 if case=='trimap_values':arr[4][0,0]=17
 if case=='no_unknown':arr[4][arr[5]]=0
 if case=='no_protected_fg':arr[1][arr[6]]=0
 for key,v in zip(['source_sha256','alpha_sha256','estimated_rgb_sha256','estimated_alpha_sha256','trimap_sha256','region_sha256','protected_sha256'],arr):k[key]=digest(v)
 with pytest.raises(ValueError):apply_reviewed_matting_patch(*arr,**k)


def test_no_mutation():
 arr,k=sample();before=[x.copy()for x in arr];apply_reviewed_matting_patch(*arr,**k);assert all(np.array_equal(x,y)for x,y in zip(arr,before))


@pytest.mark.parametrize('seed',range(12))
def test_random_native_bounds(seed):
 arr,k=sample();rng=np.random.default_rng(seed);arr[2]=rng.integers(0,256,arr[0].shape,dtype=np.uint8);arr[3]=rng.integers(0,256,arr[1].shape,dtype=np.uint8);k['estimated_rgb_sha256']=digest(arr[2]);k['estimated_alpha_sha256']=digest(arr[3]);k['feather']=8;k['support_policy']='retain';o,b,m=apply_reviewed_matting_patch(*arr,**k);assert np.array_equal(o[~arr[5]],arr[0][~arr[5]]);assert np.array_equal(b[~arr[5]],arr[1][~arr[5]]);assert np.max(np.abs(o.astype(int)-arr[0]))<=24;assert np.array_equal(b>0,arr[1]>0);assert np.array_equal(b==255,arr[1]==255)
