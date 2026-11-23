import json,math
import numpy as np
import pytest
from renderello.qa import masked_ssim
from renderello.schemas import QAReport

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_unavailable_metrics_are_null_not_nonstandard_numbers(value):
 r=QAReport('s','o','cutout_compose','clean_studio',0,ssim_product=value,l1_product=value,color_drift=value,edge_retained=value)
 text=r.to_json();decoded=json.loads(text,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
 for field in ['ssim_product','l1_product','color_drift','edge_retained']:assert decoded[field] is None
 assert math.isnan(r.ssim_product) if math.isnan(value) else r.ssim_product==value

def test_finite_metrics_unchanged_and_unrated_stays_unrated():
 r=QAReport('s','o','cutout_compose','clean_studio',0,ssim_product=1,l1_product=0,color_drift=.1,edge_retained=None)
 d=json.loads(r.to_json());assert d['ssim_product']==1 and d['l1_product']==0 and d['edge_retained'] is None and d['score_fidelity'] is None

@pytest.mark.parametrize('shape',[(5,20),(20,5),(3,30),(30,3)])
def test_ssim_small_side_uses_valid_odd_window(shape):
 a=np.full((*shape,3),120,np.uint8);m=np.full(shape,255,np.uint8)
 assert masked_ssim(a,a,m)==pytest.approx(1)

@pytest.mark.parametrize('shape',[(1,100),(2,100)])
def test_ssim_unresolvable_side_is_unavailable(shape):
 a=np.full((*shape,3),120,np.uint8);m=np.full(shape,255,np.uint8)
 assert math.isnan(masked_ssim(a,a,m))
