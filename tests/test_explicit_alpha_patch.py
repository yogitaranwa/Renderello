import numpy as np
import pytest
from PIL import Image
from renderello.explicit_alpha_patch import apply_explicit_alpha_patches as apply

@pytest.fixture
def case():
 a=np.zeros((50,60),dtype='uint8');a[8:45,10:50]=255;a[9,10:15]=128
 return Image.fromarray(a),{'label':[20,20,35,35]}

def patch(op='exclude',polygon=None,note='Source boundary visually inspected'):
 return dict(operation=op,polygon=polygon or [[10,8],[15,8],[15,12],[10,12]],source_review_note=note)

def test_no_patches_is_pixel_identical(case):
 a,p=case;b,m=apply(a,p,[]);assert a.tobytes()==b.tobytes();assert m['alpha_changed_pixels']==0;assert not m['quality_accepted']

def test_exclude_local_only_and_input_immutable(case):
 a,p=case;before=a.tobytes();b,m=apply(a,p,[patch()]);aa=np.array(a);bb=np.array(b)
 assert a.tobytes()==before;assert m['support_removed']==30;assert m['alpha_changed_pixels']==30;assert m['protected_alpha_changed']==0
 assert np.array_equal(aa[20:35,20:35],bb[20:35,20:35]);assert np.array_equal(aa[:,16:],bb[:,16:])

def test_restore_opaque_support_and_soft_pixels(case):
 a,p=case;b,m=apply(a,p,[patch('restore_opaque',[[8,8],[14,8],[14,10],[8,10]])]);assert m['support_added']==6;assert m['alpha_changed_pixels']==11;assert m['support_removed']==0;assert np.all(np.array(b)[8:11,8:15]==255)

def test_order_cannot_hide_overlapping_annotations(case):
 a,p=case
 with pytest.raises(ValueError,match='overlapping'):apply(a,p,[patch(),patch('restore_opaque')])

@pytest.mark.parametrize('op',['exclude','restore_opaque'])
def test_protected_crossing_refused_even_when_no_change(case,op):
 a,p=case
 with pytest.raises(ValueError,match='protected'):apply(a,p,[patch(op,[[20,20],[25,20],[25,25],[20,25]])])

@pytest.mark.parametrize('poly',[
 [[-1,1],[4,1],[4,4]],[[1,1],[70,1],[4,4]],[[1.5,1],[4,1],[4,4]],
 [[1,1],[2,2],[3,3]],[[1,1],[4,1]],[[1,1],[4,1],[4,4],[1,1]],
 [[1,1],[5,5],[1,5],[4,1]],[[True,1],[4,1],[4,4]],[[1,1,2],[4,1],[4,4]]])
def test_bad_polygons(case,poly):
 a,p=case
 with pytest.raises(ValueError):apply(a,p,[patch(polygon=poly)])

@pytest.mark.parametrize('note',['','   ',None,42])
def test_missing_source_review(case,note):
 a,p=case
 with pytest.raises(ValueError):apply(a,p,[patch(note=note)])

@pytest.mark.parametrize('p',[{}, {'label':[0,0,2,2]}, {'label':[20,20,61,35]}, {'label':[20.5,20,35,35]}])
def test_invalid_protection(case,p):
 a,_=case
 with pytest.raises(ValueError):apply(a,p,[])

def test_rgb_mask_refused(case):
 a,p=case
 with pytest.raises(ValueError):apply(a.convert('RGB'),p,[])

def test_bad_operation_refused(case):
 a,p=case
 with pytest.raises(ValueError):apply(a,p,[patch('automatic')])

def test_two_disjoint_patches(case):
 a,p=case;b,m=apply(a,p,[patch(),patch('restore_opaque',[[2,2],[4,2],[4,4],[2,4]])]);assert m['support_removed']==30;assert m['support_added']==9;assert len(m['patches'])==2

def test_export_never_approves_or_overwrites(case,tmp_path):
 from renderello.explicit_alpha_patch import export_explicit_alpha_review
 import json
 a,p=case;source=Image.fromarray(np.arange(50*60*3,dtype='uint16').reshape(50,60,3).astype('uint8'))
 out=tmp_path/'candidate';m=export_explicit_alpha_review(source,a,p,[patch()],out)
 assert not m['quality_accepted'];assert not m['production_review_created'];assert m['retained_opaque_rgb_changed']==0
 assert (out/'manifest.json').is_file();assert not (out/'reviewed-alpha.png').exists()
 expected=np.array(source);alpha=np.array(Image.open(out/'candidate-alpha.png'))
 for name in ['light','dark']:
  actual=np.array(Image.open(out/f'{name}-native.png'));assert np.array_equal(actual[alpha==255],expected[alpha==255])
 with pytest.raises(ValueError,match='preserve'):export_explicit_alpha_review(source,a,p,[patch()],out)
 assert json.loads((out/'manifest.json').read_text())==m

@pytest.mark.parametrize('mode,size',[('RGBA',(60,50)),('RGB',(59,50))])
def test_export_bad_source_no_files(case,tmp_path,mode,size):
 from renderello.explicit_alpha_patch import export_explicit_alpha_review
 a,p=case;out=tmp_path/'bad'
 with pytest.raises(ValueError):export_explicit_alpha_review(Image.new(mode,size),a,p,[patch()],out)
 assert not out.exists()

def test_export_invalid_patch_no_files(case,tmp_path):
 from renderello.explicit_alpha_patch import export_explicit_alpha_review
 a,p=case;out=tmp_path/'bad'
 with pytest.raises(ValueError):export_explicit_alpha_review(Image.new('RGB',a.size),a,p,[patch(note='')],out)
 assert not out.exists()
