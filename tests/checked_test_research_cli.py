import unittest,tempfile,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from renderello.checked_research.research_cli import parser,execute

class T(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.r=Path(self.t.name)
  Image.new('RGB',(12,10),(20,70,110)).save(self.r/'s.png');Image.new('RGB',(12,10),(40,100,140)).save(self.r/'raw.png');Image.new('L',(12,10),255).save(self.r/'a.png')
 def tearDown(self):self.t.cleanup()
 def args(self):
  a=['lighting','--output-dir',str(self.r/'out'),'--registration-reviewed']
  for k,n in [('source','s.png'),('raw','raw.png'),('alpha','a.png')]:
   p=self.r/n;a+=['--'+k,str(p),'--'+k+'-sha256',hashlib.sha256(p.read_bytes()).hexdigest()]
  return parser().parse_args(a)
 def test_manifest_and_files(self):
  m=execute(self.args());self.assertFalse(m['publication_accepted']);self.assertIsNone(m['quality_accuracy']);self.assertTrue((self.r/'out/manifest.json').is_file())
  for n,r in m['outputs'].items():self.assertEqual(hashlib.sha256((self.r/'out'/n).read_bytes()).hexdigest(),r['sha256'])
 def test_no_overwrite(self):
  a=self.args();execute(a);before=(self.r/'out/output.png').read_bytes()
  with self.assertRaises(ValueError):execute(a)
  self.assertEqual(before,(self.r/'out/output.png').read_bytes())
 def test_bad_raw_binding(self):
  a=self.args();a.raw_sha256='bad'
  with self.assertRaises(ValueError):execute(a)
  self.assertFalse((self.r/'out').exists())
 def test_bad_alpha_binding(self):
  a=self.args();a.alpha_sha256='bad'
  with self.assertRaises(ValueError):execute(a)
  self.assertFalse((self.r/'out').exists())
 def test_no_alignment_declaration(self):
  a=self.args();a.registration_reviewed=False
  with self.assertRaises(ValueError):execute(a)
  self.assertFalse((self.r/'out').exists())
 def test_wrong_native_shape(self):
  Image.new('RGB',(6,5)).save(self.r/'raw.png');a=self.args()
  with self.assertRaises(ValueError):execute(a)
  self.assertFalse((self.r/'out').exists())
if __name__=='__main__':unittest.main()
