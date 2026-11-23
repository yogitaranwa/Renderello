import numpy as np
import pytest
from renderello.scene_finish import inspect_support_components


def test_single_connected_body_with_two_feet():
 a=np.zeros((80,80),np.uint8);a[10:45,10:70]=255;a[45:60,10:30]=255;a[45:60,50:70]=255
 m=inspect_support_components(a)
 assert m['component_count']==1 and m['largest_component_bottom']==59

@pytest.mark.parametrize('size',[1,4,12])
def test_lower_detached_island_rejected_without_modifying_alpha(size):
 a=np.zeros((100,100),np.uint8);a[5:65,25:70]=255;a[85:85+size,5:5+size]=255;before=a.copy()
 with pytest.raises(ValueError,match='detached lower'):inspect_support_components(a)
 assert np.array_equal(a,before)

def test_same_baseline_disconnected_shapes_not_semantically_approved():
 a=np.zeros((80,80),np.uint8);a[10:60,10:40]=255;a[30:60,50:70]=255
 m=inspect_support_components(a)
 assert m['component_count']==2 and m['mask_semantics']=='NOT_EVALUATED'


def test_composition_rejects_lower_island_before_output_creation(tmp_path):
 from PIL import Image,ImageDraw
 from renderello.reviewed_source import prepare_reviewed_source
 from renderello.scene_plate import compose_plate
 src=Image.new('RGB',(100,120),'white');a=Image.new('L',src.size,0)
 ImageDraw.Draw(src).rectangle((30,10,75,85),fill='red');d=ImageDraw.Draw(a);d.rectangle((30,10,75,85),fill=255);d.rectangle((5,105,12,110),fill=255)
 src.save(tmp_path/'src.png');a.save(tmp_path/'a.png')
 prepare_reviewed_source(tmp_path/'src.png',tmp_path/'prepared',mask=tmp_path/'a.png',mask_coordinates='display')
 Image.new('RGB',(1024,1024),'gray').save(tmp_path/'plate.png')
 with pytest.raises(ValueError,match='detached lower'):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'out',baseline=900)
 assert not (tmp_path/'out').exists()
