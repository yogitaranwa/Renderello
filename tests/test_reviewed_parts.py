import numpy as np
import pytest
from renderello.reviewed_parts import audit_parts
from renderello.candidate_diagnostics import digest

@pytest.fixture
def case():
 rgb=np.full((20,20,3),88,np.uint8);a=np.zeros((20,20),np.uint8);a[2:18,3:17]=255;a[17,3:17]=120
 m=np.zeros(a.shape,bool);m[15:]=True
 return [rgb,a,rgb.copy(),a.copy(),[dict(name='hem',mask=m,mask_sha256=digest(m))]],dict(source_sha256=digest(rgb),prior_alpha_sha256=digest(a),reviewed=True,review_note='Synthetic region not independent truth.')

def test_exact(case):
 args,kw=case;r=audit_parts(*args,**kw);assert r['all_named_parts_preserved'] and not r['accepted'] and not r['independent_ground_truth']

def test_fractional_loss(case):
 args,kw=case;args[3][17,3:17]=0;r=audit_parts(*args,**kw)['regions'][0]
 assert r['alpha_loss_units']==1680 and r['prior_covered_pixels_lost']==14 and not r['exact_prior_preservation']

def test_rgb_hidden_by_alpha_still_fails(case):
 args,kw=case;args[2][19,0]=9;r=audit_parts(*args,**kw);assert not r['all_named_parts_preserved']

@pytest.mark.parametrize('kind',['hash','priorhash','regionhash','floatmask','empty','shape','duplicate','review','note','no_foreground','rgbshape','alphatype','too_many'])
def test_refusals(case,kind):
 args,kw=case
 if kind=='hash':kw['source_sha256']='bad'
 elif kind=='priorhash':kw['prior_alpha_sha256']='bad'
 elif kind=='regionhash':args[4][0]['mask_sha256']='bad'
 elif kind=='floatmask':args[4][0]['mask']=args[4][0]['mask'].astype(float)
 elif kind=='empty':args[4]=[]
 elif kind=='shape':args[4][0]['mask']=np.ones((3,3),bool)
 elif kind=='duplicate':args[4]*=2
 elif kind=='review':kw['reviewed']=False
 elif kind=='note':kw['review_note']=' '
 elif kind=='no_foreground':
  m=np.zeros((20,20),bool);m[19]=True;args[4][0].update(mask=m,mask_sha256=digest(m))
 elif kind=='rgbshape':args[2]=args[2][:-1]
 elif kind=='alphatype':args[3]=args[3].astype(float)
 elif kind=='too_many':args[4]*=33
 with pytest.raises(ValueError):audit_parts(*args,**kw)
