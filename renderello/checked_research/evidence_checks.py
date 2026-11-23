"""Fail-closed research accounting. File validity and error metrics are not photo quality."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def image_record(path,expected_sha256=None):
 p=Path(path);digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('image hash mismatch')
 with Image.open(p) as im:im.verify()
 with Image.open(p) as im:
  rgb=np.asarray(im.convert('RGB'));size=list(im.size)
 return dict(path=str(p),sha256=digest,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),size=size)

def source_bound_alpha(source,alpha,expected_source_sha256,*,expected_alpha_sha256=None):
 rec=image_record(source,expected_source_sha256);p=Path(alpha)
 if expected_alpha_sha256 is not None and hashlib.sha256(p.read_bytes()).hexdigest()!=expected_alpha_sha256:raise ValueError('alpha hash mismatch')
 with Image.open(p) as a:
  if a.mode!='L' or list(a.size)!=rec['size']:raise ValueError('source-native L alpha required; no automatic resize')
  arr=np.asarray(a)
 if not arr.any():raise ValueError('empty alpha')
 return dict(source=rec,alpha_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),support_pixels=int((arr>0).sum()),opaque_pixels=int((arr==255).sum()),components_reviewed=False,boundary_accuracy=None)

def correspondence_metrics(pred,truth,valid):
 if not isinstance(pred,np.ndarray) or pred.ndim!=3 or pred.shape[2]!=2 or truth.shape!=pred.shape:raise ValueError('matching HxWx2 arrays required')
 if pred.dtype.kind!='f' or truth.dtype.kind!='f' or not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('finite floating correspondence required')
 h,w=pred.shape[:2]
 if min(h,w)<2 or valid.dtype!=np.bool_ or valid.shape!=(h,w):raise ValueError('matching boolean valid mask and dimensions>=2 required')
 inbounds=((truth>=0)&(truth<=1)).all(2);v=valid&inbounds
 if not v.any():raise ValueError('no in-bounds valid pixels')
 err=np.linalg.norm((pred-truth)*np.array([w-1,h-1]),axis=2)[v]
 return dict(valid_pixels=int(v.sum()),excluded_truth_out_of_bounds=int((valid&~inbounds).sum()),pred_out_of_bounds_valid=int((v&~((pred>=0)&(pred<=1)).all(2)).sum()),mean_epe_pixels=float(err.mean()),median_epe_pixels=float(np.median(err)),over3px_fraction=float((err>3).mean()),over5px_fraction=float((err>5).mean()),over10px_fraction=float((err>10).mean()),product_accuracy=None,publication_gate=None)

def acceptance_summary(rows):
 # Software cannot establish rater identity or blinding from imported text.
 if not isinstance(rows,list):raise ValueError('list of review rows required')
 dimensions=['fidelity','photorealism','contact','material_edges','composition','distinctness','publish_usefulness'];seen=set()
 for row in rows:
  k=(row.get('source_sha256'),row.get('output_sha256'),row.get('rater_id'))
  if not all(isinstance(v,str) and v for v in k):raise ValueError('source/output hashes and rater identity declarations required')
  if k in seen:raise ValueError('duplicate review row')
  seen.add(k)
  if any(isinstance(row.get(d),bool) or not isinstance(row.get(d),int) or not 1<=row[d]<=5 for d in dimensions):raise ValueError('all seven integer1..5scores required')
  if type(row.get('critical_error')) is not bool:raise ValueError('explicit critical-error decision required')
 return dict(imported_rows=len(rows),no_critical_alteration_fraction=None,publish_ready_fraction=None,acceptance_status='NOT_ESTABLISHED',reason='Imported declarations do not authenticate rater independence/blinding or matched template and source coverage.')
