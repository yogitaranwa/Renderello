import numpy as np
import pytest
from PIL import Image
from renderello.plain_backdrop_preview import plain_backdrop_alpha

def ring():
    a=np.full((100,100,3),255,dtype='uint8');a[15:85,15:85]=[40,60,80];a[35:65,35:65]=255
    return Image.fromarray(a)

def test_enclosed_background_is_not_guessed():
    alpha,_=plain_backdrop_alpha(ring(),material='opaque')
    assert np.array(alpha)[50,50]==255

def test_explicit_background_seed_clears_entire_hole_preserves_ring():
    alpha,m=plain_backdrop_alpha(ring(),material='opaque',background_points=[(50,50)],protected_regions={'body':(20,20,30,80)})
    a=np.array(alpha);assert np.all(a[35:65,35:65]==0);assert np.all(a[20:30,20:80]==255)
    assert m['background_points']==[[50,50]] and m['seed_origin'].startswith('EXPLICIT')

def test_product_colour_seed_refused():
    with pytest.raises(ValueError,match='backdrop colour'):plain_backdrop_alpha(ring(),material='opaque',background_points=[(20,20)])

def test_conflict_with_protected_box_fails_instead_of_repainting():
    with pytest.raises(ValueError,match='crosses protected'):plain_backdrop_alpha(ring(),material='opaque',background_points=[(50,50)],protected_regions={'white_label':(40,40,60,60)})

@pytest.mark.parametrize('points',[[(-1,0)],[(100,1)],[(1.0,1)],[(True,1)],[(1,)],[(1,2,3)],[(0,0)]*65,None])
def test_malformed_or_excessive_seeds_refused(points):
    with pytest.raises(ValueError):plain_backdrop_alpha(ring(),material='opaque',background_points=points)

@pytest.mark.parametrize('regions',[{'x':(0,0,101,10)},{'x':(1,1,1,5)},{'x':(1.,1,2,5)},[],{'':(20,20,30,30)}])
def test_invalid_protected_box_refused(regions):
    with pytest.raises(ValueError):plain_backdrop_alpha(ring(),material='opaque',protected_regions=regions)

def test_exterior_seed_is_idempotent():
    a,_=plain_backdrop_alpha(ring(),material='opaque');b,_=plain_backdrop_alpha(ring(),material='opaque',background_points=[(0,0),(0,0)])
    assert np.array_equal(a,b)
