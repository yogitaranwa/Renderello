import hashlib
import numpy as np
import pytest
from renderello.reviewed_white_background import outer_white_background_alpha

def fixture():
 a=np.full((40,40,3),255,np.uint8);a[5:35,5:35]=25;a[12:18,12:18]=255
 return a

def test_outer_white_removed_enclosed_white_text_retained_rgb_untouched():
 a=fixture();b=a.copy();alpha,r=outer_white_background_alpha(a,reviewed_source_sha256=hashlib.sha256(a.tobytes()).hexdigest())
 assert alpha[0,0]==0 and alpha[8,8]==255 and alpha[14,14]==255
 assert np.array_equal(a,b) and not r['accepted']
 assert r['semantic_completeness']=='NOT_AUTOMATICALLY_EVALUATED'

def test_other_source_refused():
 with pytest.raises(ValueError,match='different source'):
  outer_white_background_alpha(fixture(),reviewed_source_sha256='0'*64)

def test_declared_thresholds_bounded():
 a=fixture()
 with pytest.raises(ValueError,match='brightness'):
  outer_white_background_alpha(a,reviewed_source_sha256=hashlib.sha256(a.tobytes()).hexdigest(),minimum_channel=100)
