import numpy as np
import pytest
from renderello.native_pixel_audit import audit_native_pixels


def data():
 s=np.arange(8*9*3,dtype=np.uint8).reshape(8,9,3)
 a=np.zeros((8,9),dtype=np.uint8);a[2:6,3:7]=255;a[2,3]=254
 o=np.zeros((12,13,3),dtype=np.uint8);o[5:9,6:10]=s[2:6,3:7]
 return s,a,o


def call(s,a,o):return audit_native_pixels(s,a,o,(3,2,4,4),(6,5,4,4))


def test_native_unresampled_exact_and_no_input_writes():
 s,a,o=data();before=[x.copy() for x in (s,a,o)];r=call(s,a,o)
 assert r['status']=='MATCH' and r['native_rgb_exact'] is True and r['fully_opaque_source_pixels']==15
 assert r['max_channel_error']==0 and r['changed_pixels']==0
 assert all(np.array_equal(x,y) for x,y in zip((s,a,o),before))
 assert r['quality_acceptance']=='NOT_EVALUATED' and r['edges']=='NOT_EVALUATED'


def test_one_channel_one_pixel_error_is_not_a_pass():
 s,a,o=data();o[6,7,1]^=1;r=call(s,a,o)
 assert r['status']=='MISMATCH' and r['native_rgb_exact'] is False
 assert r['changed_pixels']==1 and r['max_channel_error']==1


def test_nonopaque_rgb_and_background_not_claimed():
 s,a,o=data();o[5,6]=0;o[0,0]=255
 assert call(s,a,o)['native_rgb_exact'] is True


def test_no_opaque_pixels_is_unavailable_not_vacuous_pass():
 s,a,o=data();a[:]=254;r=call(s,a,o)
 assert r['status']=='NO_FULLY_OPAQUE_PIXELS' and r['native_rgb_exact'] is None and r['changed_pixels'] is None


def test_resizing_is_unavailable_even_for_identical_solid_colours():
 s,a,o=data();s[:]=50;o[:]=50
 r=audit_native_pixels(s,a,o,(3,2,4,4),(6,5,3,4))
 assert r['status']=='NATIVE_SIZE_CHANGED' and r['native_rgb_exact'] is None


@pytest.mark.parametrize('box',[(True,2,4,4),(-1,2,4,4),(3,2,0,4),(6,2,4,4),(3,2,4,7),(3.,2,4,4),(1,2,3)])
def test_bad_source_box_refused(box):
 s,a,o=data()
 with pytest.raises(ValueError):audit_native_pixels(s,a,o,box,(6,5,4,4))


@pytest.mark.parametrize('box',[(True,5,4,4),(6,5,10,4),(6,5,4,10),(-1,5,4,4)])
def test_bad_output_box_refused(box):
 s,a,o=data()
 with pytest.raises(ValueError):audit_native_pixels(s,a,o,(3,2,4,4),box)


@pytest.mark.parametrize('which',['source-float','source-rgba','alpha-float','alpha-size','output-gray','output-empty'])
def test_ambiguous_array_contract_refused(which):
 s,a,o=data()
 if which=='source-float':s=s.astype(float)
 if which=='source-rgba':s=np.zeros((8,9,4),dtype=np.uint8)
 if which=='alpha-float':a=a.astype(float)
 if which=='alpha-size':a=a[:1]
 if which=='output-gray':o=o[:,:,0]
 if which=='output-empty':o=o[:0]
 with pytest.raises(ValueError):call(s,a,o)


def test_subtraction_does_not_wrap_uint8():
 s,a,o=data();s[:]=255;o[:]=0;r=call(s,a,o)
 assert r['max_channel_error']==255 and r['changed_pixels']==15
