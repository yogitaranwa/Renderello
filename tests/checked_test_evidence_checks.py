import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from renderello.checked_research.evidence_checks import *
class T(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name);Image.new('RGB',(16,12),'red').save(self.p/'s.png');Image.new('L',(16,12),255).save(self.p/'a.png');self.sha=image_record(self.p/'s.png')['sha256'];self.f=np.zeros((12,16,2),float);self.m=np.ones((12,16),bool)
 def tearDown(self):self.tmp.cleanup()
 def test_image_hash(self):
  with self.assertRaises(ValueError):image_record(self.p/'s.png','bad')
 def test_image_corrupt(self):
  (self.p/'bad').write_bytes(b'notimage')
  with self.assertRaises(Exception):image_record(self.p/'bad')
 def test_alpha_bound(self):self.assertEqual(source_bound_alpha(self.p/'s.png',self.p/'a.png',self.sha)['opaque_pixels'],192)
 def test_alpha_dimensions(self):
  Image.new('L',(15,12),255).save(self.p/'a.png')
  with self.assertRaises(ValueError):source_bound_alpha(self.p/'s.png',self.p/'a.png',self.sha)
 def test_alpha_mode(self):
  Image.new('RGB',(16,12),'white').save(self.p/'a.png')
  with self.assertRaises(ValueError):source_bound_alpha(self.p/'s.png',self.p/'a.png',self.sha)
 def test_alpha_empty(self):
  Image.new('L',(16,12),0).save(self.p/'a.png')
  with self.assertRaises(ValueError):source_bound_alpha(self.p/'s.png',self.p/'a.png',self.sha)
 def test_alpha_hash(self):
  with self.assertRaises(ValueError):source_bound_alpha(self.p/'s.png',self.p/'a.png',self.sha,expected_alpha_sha256='bad')
 def test_flow_identity(self):self.assertEqual(correspondence_metrics(self.f,self.f,self.m)['mean_epe_pixels'],0)
 def test_flow_pixelunits(self):
  q=self.f.copy();q[:,:,0]=1/15;self.assertAlmostEqual(correspondence_metrics(q,self.f,self.m)['mean_epe_pixels'],1)
 def test_flow_nan(self):
  self.f[0,0]=np.nan
  with self.assertRaises(ValueError):correspondence_metrics(self.f,self.f,self.m)
 def test_flow_mask(self):
  with self.assertRaises(ValueError):correspondence_metrics(self.f,self.f,self.m.astype(float))
 def test_flow_empty(self):
  with self.assertRaises(ValueError):correspondence_metrics(self.f,self.f,np.zeros_like(self.m))
 def test_flow_bounds(self):
  q=self.f.copy();q[0,0]=2;self.assertEqual(correspondence_metrics(self.f,q,self.m)['excluded_truth_out_of_bounds'],1)
 def test_flow_shapes(self):
  with self.assertRaises(ValueError):correspondence_metrics(self.f[:1],self.f,self.m)
 def test_empty_ratings(self):self.assertIsNone(acceptance_summary([])['publish_ready_fraction'])
 def test_bad_ratings(self):
  with self.assertRaises(ValueError):acceptance_summary([{}])
 def test_duplicate_rating(self):
  r=dict(source_sha256='a',output_sha256='b',rater_id='self',critical_error=False,**{k:4 for k in ['fidelity','photorealism','contact','material_edges','composition','distinctness','publish_usefulness']})
  with self.assertRaises(ValueError):acceptance_summary([r,r])
 def test_no_auto_acceptance(self):
  r=dict(source_sha256='a',output_sha256='b',rater_id='self',critical_error=False,**{k:5 for k in ['fidelity','photorealism','contact','material_edges','composition','distinctness','publish_usefulness']});self.assertIsNone(acceptance_summary([r])['publish_ready_fraction'])
if __name__=='__main__':unittest.main()
