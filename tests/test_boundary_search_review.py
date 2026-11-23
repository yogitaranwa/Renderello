import numpy as np
import pytest
from renderello.boundary_search_review import boundary_search_review


def test_limit_report_never_accepts():
 r=boundary_search_review([10,11,12],[12,11,10],corridor_radius=2)
 assert r['refusal_flags']==['SEARCH_LIMIT_REACHED']
 assert r['corridor_limit_column_indices']==[0,2]
 assert not r['quality_accepted'] and not r['production_mask_selected']


def test_unflagged_is_not_acceptance():
 r=boundary_search_review([10,11],[10,12],corridor_radius=2)
 assert r['refusal_flags']==[]
 assert r['semantic_review_required'] and not r['quality_accepted']


@pytest.mark.parametrize('b,t,r',[
 ([1,2],[1],2),([],[],2),([1.0,2.0],[1,2],2),
 ([True,False],[1,2],2),([1,2],[1,2],True),
 ([1,2],[1,2],0),([1,2],[1,2],129),
 ([-1,2],[1,2],2),([1,2],[1,1000001],2),
 ([1,2],[1,5],2),([[1,2]],[[1,2]],2),
 (np.array([1,2],dtype='uint64'),np.array([2**64-1,2],dtype='uint64'),2)])
def test_invalid_geometry_refused(b,t,r):
 with pytest.raises(ValueError):boundary_search_review(b,t,corridor_radius=r)
