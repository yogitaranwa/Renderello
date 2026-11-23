import unittest
import numpy as np
from renderello.checked_research.conditioning_canvas import conditioning_canvas

class CanvasTests(unittest.TestCase):
 def setUp(self):
  self.s=np.arange(64*48*3,dtype=np.uint8).reshape(64,48,3);self.m=np.zeros((64,48),np.uint8);self.m[4:60,6:42]=255
 def call(self,**kw):return conditioning_canvas(self.s,self.m,size=(96,96),**kw)
 def test_native_exact_center(self):
  c,m,d=self.call();self.assertTrue(np.array_equal(c[16:80,24:72],self.s));self.assertTrue(np.array_equal(m[16:80,24:72],self.m));self.assertEqual(d['source_rectangle'],[24,16,72,80]);self.assertFalse(d['quality_accepted'])
 def test_no_input_mutation(self):
  s=self.s.copy();m=self.m.copy();c,a,_=self.call();c[:]=0;a[:]=0;self.assertTrue(np.array_equal(s,self.s));self.assertTrue(np.array_equal(m,self.m))
 def test_custom_offset_padding(self):
  c,m,d=self.call(offset=(0,0),padding=(1,2,3));self.assertEqual(c[-1,-1].tolist(),[1,2,3]);self.assertEqual(m[-1,-1],0);self.assertEqual(d['source_rectangle'],[0,0,48,64])
 def test_no_resize(self):
  with self.assertRaises(ValueError):conditioning_canvas(self.s,self.m,size=(32,32))
 def test_dimensions(self):
  for sz in [(0,96),(95,96),(True,96),(32,128000),(96,),None]:
   with self.assertRaises(ValueError):conditioning_canvas(self.s,self.m,size=sz)
 def test_offsets(self):
  for o in [(-1,0),(49,0),(0,33),(True,0),(0.0,0),(1,),{}]:
   with self.assertRaises(ValueError):self.call(offset=o)
 def test_padding(self):
  for p in [(0,0,256),(-1,0,0),(True,0,0),(0,0),(0.,0,0),'gray']:
   with self.assertRaises(ValueError):self.call(padding=p)
 def test_mask_dtype(self):
  with self.assertRaises(ValueError):conditioning_canvas(self.s,self.m.astype(float))
 def test_mask_shape(self):
  with self.assertRaises(ValueError):conditioning_canvas(self.s,self.m[:1])
 def test_source_dtype(self):
  with self.assertRaises(ValueError):conditioning_canvas(self.s.astype(float),self.m)
 def test_empty_mask(self):
  with self.assertRaises(ValueError):conditioning_canvas(self.s,np.zeros_like(self.m))
 def test_soft_mask_unchanged(self):
  self.m[20,20]=37;c,m,d=self.call();self.assertEqual(m[36,44],37)
 def test_hash_binding(self):
  d=self.call()[2];self.s[2,2,0]^=1;self.assertNotEqual(d['source_pixels_sha256'],self.call()[2]['source_pixels_sha256'])
 def test_occupancy_is_support_only(self):
  d=self.call()[2];self.assertAlmostEqual(d['protection_height_fraction'],56/96);self.assertEqual(d['semantic_mask_accuracy'],'NOT_EVALUATED');self.assertFalse(d['old_background_removed'])
 def test_full_frame_mask_not_semantic_pass(self):
  self.m[:]=255;c,m,d=self.call();self.assertEqual(d['protection_support_box'],[24,16,72,80]);self.assertFalse(d['quality_accepted'])

if __name__=='__main__':unittest.main()
