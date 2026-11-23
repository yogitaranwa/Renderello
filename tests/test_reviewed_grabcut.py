from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pytest
from renderello.checked_research.reviewed_grabcut import seed_map

def sample():
 return dict(source_sha256=hashlib.sha256(b'x').hexdigest(),source_size=[20,20],origin='assistant visual review coarse manual polygons, not ground truth',annotation_coordinate_space=[20,20],probable_foreground=[[[2,2],[17,2],[17,17],[2,17]]],definite_foreground=[[[5,5],[14,5],[14,14],[5,14]]],definite_background=[[[0,0],[1,0],[1,19],[0,19]]])

def test_native_class_map():
 a=seed_map(sample(),b'x',(20,20));assert a.shape==(20,20) and a[8,8]==1 and a[3,3]==3 and a[0,0]==0

@pytest.mark.parametrize('change',[{'source_sha256':'bad'},{'source_size':[21,20]},{'origin':'automatic'},{'annotation_coordinate_space':[20,19]},{'definite_foreground':[]},{'probable_foreground':[[[0,0],[20,0],[20,20]]]},{'definite_foreground':[[[1,1],[1,1],[1,1]]]},{'definite_background':[[[0,0],[float('nan'),1],[2,2]]]}])
def test_reject_bad_review(change):
 s=sample();s.update(change)
 with pytest.raises(ValueError):seed_map(s,b'x',(20,20))
