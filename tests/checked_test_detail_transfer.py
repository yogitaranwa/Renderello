import unittest,numpy as np
from renderello.checked_research.detail_transfer import transfer
class T(unittest.TestCase):
 def setUp(self):self.s=np.full((20,24,3),100,np.uint8);self.r=np.full_like(self.s,220);self.a=np.full((20,24),255,np.uint8)
 def run_t(self,**kw):return transfer(self.s,self.r,self.a,registration_reviewed=True,**kw)
 def test_registration(self):
  with self.assertRaises(ValueError):transfer(self.s,self.r,self.a)
 def test_shape(self):
  with self.assertRaises(ValueError):transfer(self.s,self.r[:1],self.a,registration_reviewed=True)
 def test_dtype(self):
  with self.assertRaises(ValueError):transfer(self.s.astype(float),self.r,self.a,registration_reviewed=True)
 def test_alpha(self):
  with self.assertRaises(ValueError):transfer(self.s,self.r,self.a.astype(float),registration_reviewed=True)
 def test_empty(self):
  with self.assertRaises(ValueError):transfer(self.s,self.r,np.zeros_like(self.a),registration_reviewed=True)
 def test_sigma(self):
  for v in (0,-1,101,float('nan')):
   with self.assertRaises(ValueError):self.run_t(sigma=v)
 def test_gain(self):
  for v in ((1.1,1.5),(.5,.9),(.5,3),(float('nan'),1.5)):
   with self.assertRaises(ValueError):self.run_t(gain_bounds=v)
 def test_identity(self):
  out,_,m=transfer(self.s,self.s,self.a,registration_reviewed=True);self.assertTrue(np.array_equal(out,self.s));self.assertFalse(m['publication_accepted'])
 def test_transparent_background(self):
  self.a[:5]=0;out,_,_=self.run_t();self.assertTrue(np.array_equal(out[:5],self.r[:5]))
 def test_no_new_clip(self):
  self.s[:]=(250,100,50);out,g,m=self.run_t();self.assertEqual(m['preencode_out_of_gamut_pixels'],0);self.assertGreater(m['highlight_capped_pixels'],0)
 def test_deterministic(self):self.assertTrue(np.array_equal(self.run_t()[0],self.run_t()[0]))
 def test_not_source_exact(self):self.assertGreater(self.run_t()[2]['opaque_rgb_changed'],0)
 def test_dark_textured_identity(self):
  rng=np.random.default_rng(12);self.s=rng.integers(0,20,self.s.shape,dtype=np.uint8);out,_,_=transfer(self.s,self.s,self.a,registration_reviewed=True);self.assertTrue(np.array_equal(out,self.s))
 def test_black(self):
  self.s[:]=0;out,g,m=self.run_t();self.assertTrue(np.isfinite(g).all());self.assertEqual(m['preencode_out_of_gamut_pixels'],0)
if __name__=='__main__':unittest.main()
