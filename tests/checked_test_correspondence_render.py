import unittest
import numpy as np
from renderello.checked_research.correspondence_render import render_correspondence as render

class T(unittest.TestCase):
 def setUp(self):
  self.bg=np.arange(36,dtype=np.uint8).reshape(3,4,3)*7
  yy,xx=np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,4),indexing='ij');self.c=np.stack([xx,yy],2);self.v=np.ones((3,4),bool)
 def test_identity(self):
  rgb,a,m=render(self.bg,self.c,self.v);np.testing.assert_array_equal(rgb,self.bg);self.assertTrue((a==255).all());self.assertEqual(m['border_clamped_pixels'],0)
 def test_center_bilinear(self):
  c=np.full((2,2,2),.5);rgb,_,_=render(self.bg,c,np.ones((2,2),bool));np.testing.assert_array_equal(rgb[0,0],np.rint((self.bg[1,1].astype(float)+self.bg[1,2])/2).astype('uint8'))
 def test_reject(self):
  self.c[0,0,0]=-0.1
  with self.assertRaisesRegex(ValueError,'1 requested'):render(self.bg,self.c,self.v)
 def test_mask(self):
  self.c[0,0,0]=-0.1;rgb,a,m=render(self.bg,self.c,self.v,out_of_image='mask');self.assertEqual(a[0,0],0);self.assertTrue((rgb[0,0]==0).all());self.assertEqual(m['masked_out_of_image_pixels'],1)
 def test_border(self):
  self.c[0,0,0]=-0.1;rgb,a,m=render(self.bg,self.c,self.v,out_of_image='border');np.testing.assert_array_equal(rgb[0,0],self.bg[0,0]);self.assertEqual(a[0,0],255);self.assertEqual(m['border_clamped_pixels'],1)
 def test_outside_invalid_ignored(self):
  self.c[0,0]=-1;self.v[0,0]=False;_,a,m=render(self.bg,self.c,self.v);self.assertEqual(a[0,0],0);self.assertEqual(m['out_of_image_requested_pixels'],0)
 def test_nonfinite_even_invalid_refused(self):
  self.c[0,0]=np.nan;self.v[0,0]=False
  with self.assertRaises(ValueError):render(self.bg,self.c,self.v)
 def test_invalid_types(self):
  for bg,c,v in [(None,self.c,self.v),(self.bg,None,self.v),(self.bg,self.c,None),(self.bg,self.c,self.v.astype('uint8')),(self.bg,self.c.astype(int),self.v)]:
   with self.assertRaises(ValueError):render(bg,c,v)
 def test_empty_mask(self):
  with self.assertRaises(ValueError):render(self.bg,self.c,np.zeros_like(self.v))
 def test_all_outside_mask(self):
  with self.assertRaises(ValueError):render(self.bg,self.c+2,self.v,out_of_image='mask')
 def test_unknown_policy(self):
  with self.assertRaises(ValueError):render(self.bg,self.c,self.v,out_of_image='clip')
 def test_input_unchanged(self):
  bg=self.bg.copy();c=self.c.copy();v=self.v.copy();render(self.bg,self.c,self.v);np.testing.assert_array_equal(bg,self.bg);np.testing.assert_array_equal(c,self.c);np.testing.assert_array_equal(v,self.v)
 def test_nonsquare_background(self):
  c=np.zeros((2,2,2));c[1,1]=1;rgb,_,_=render(self.bg,c,np.ones((2,2),bool));np.testing.assert_array_equal(rgb[1,1],self.bg[-1,-1])

if __name__=='__main__':unittest.main()
