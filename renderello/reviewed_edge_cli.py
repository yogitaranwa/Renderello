"""File-bound opt-in edge gain/lift replay. No inferred annotation or quality promotion."""
from pathlib import Path
from io import BytesIO
import argparse,hashlib,json,os,shutil,tempfile
import numpy as np
from PIL import Image
from .image_intake import _read_bounded
from .candidate_diagnostics import digest
from .reviewed_edge_gain import darken_reviewed_edge
from .reviewed_edge_lift import lift_reviewed_edge

NAMES=('source','alpha','target','donor','protected')
def sha(data):return hashlib.sha256(data).hexdigest()
def _json_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON field: '+k)
  d[k]=v
 return d

def load_spec(path,expected):
 raw=_read_bounded(Path(path),65536)
 if sha(raw)!=expected:raise ValueError('spec file hash mismatch')
 try:s=json.loads(raw,object_pairs_hook=_json_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
 except (json.JSONDecodeError,UnicodeDecodeError) as e:raise ValueError('invalid spec JSON') from e
 fields={'schema','operation','files','reviewed','material','review_note','controls'}
 if not isinstance(s,dict) or set(s)!=fields or s['schema']!='reviewed-edge-files.v1':raise ValueError('exact spec schema required')
 if s['operation'] not in ('darken','lift'):raise ValueError('explicit darken/lift operation required')
 if not isinstance(s['files'],dict) or set(s['files'])!=set(NAMES):raise ValueError('all five file bindings required')
 for h in s['files'].values():
  if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('lowercase file SHA256 required')
 shared={'edge_width','donor_distance','strength','max_code_delta'};keys=shared|({'minimum_gain'} if s['operation']=='darken' else {'maximum_gain'})
 if not isinstance(s['controls'],dict) or set(s['controls'])!=keys:raise ValueError('exact explicit operation controls required')
 return s

def read_image(path,expected,name):
 raw=_read_bounded(Path(path),8_000_000)
 if sha(raw)!=expected:raise ValueError(name+' file hash mismatch')
 # Explicit 8-bit PNG avoids hidden color conversion, high-depth quantization or orientation.
 if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw)<29 or raw[24]!=8:raise ValueError(name+' must be8bit PNG')
 with Image.open(BytesIO(raw)) as im:
  if im.format!='PNG' or im.mode!=('RGB' if name=='source' else 'L'):raise ValueError(name+' exact RGB/L mode required')
  if getattr(im,'n_frames',1)!=1 or im.getexif() or any(k in im.info for k in ('icc_profile','gamma','srgb','transparency')):raise ValueError(name+' color/orientation/frame metadata requires separate intake')
  if min(im.size)<2 or im.width*im.height>1_000_000:raise ValueError(name+' native1MP limit; no resize')
  a=np.array(im)
 if name not in ('source','alpha'):
  if not np.isin(a,[0,255]).all():raise ValueError(name+' mask must use only0/255; no threshold guessing')
  a=a==255
 return a,dict(file_sha256=sha(raw),pixel_sha256=digest(a),bytes=len(raw),shape=list(a.shape))

def execute(*,source,alpha,target,donor,protected,spec,spec_sha256,output_dir):
 dest=Path(output_dir)
 if dest.exists() or dest.is_symlink():raise ValueError('output exists; preserve earlier evidence')
 s=load_spec(spec,spec_sha256);arrays=[];records={}
 for n,p in zip(NAMES,(source,alpha,target,donor,protected)):
  a,r=read_image(p,s['files'][n],n);arrays.append(a);records[n]=r
 rgb,al,t,d,p=arrays;before=[v.copy() for v in arrays]
 f=darken_reviewed_edge if s['operation']=='darken' else lift_reviewed_edge
 out,metrics=f(*arrays,source_sha256=digest(rgb),alpha_sha256=digest(al),target_sha256=digest(t),donor_sha256=digest(d),protected_sha256=digest(p),reviewed=s['reviewed'],material=s['material'],review_note=s['review_note'],**s['controls'])
 if any(not np.array_equal(a,b) for a,b in zip(arrays,before)):raise RuntimeError('input mutated')
 if not np.array_equal(out[p],rgb[p]) or not np.array_equal(out[~t],rgb[~t]):raise RuntimeError('protected/outside RGB changed')
 # Validation and computation precede reservation. Stage first; reserve with exclusive mkdir.
 dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.edge-stage-',dir=dest.parent));reserved=False
 try:
  Image.fromarray(out).save(temp/'foreground.png');Image.fromarray(al).save(temp/'alpha.png');Image.fromarray(np.dstack([out,al])).save(temp/'component-rgba.png')
  outputs={q.name:sha(q.read_bytes()) for q in sorted(temp.iterdir())}
  manifest=dict(schema='reviewed-edge-file-result.v1',spec_sha256=spec_sha256,operation=s['operation'],inputs=records,controls=s['controls'],metrics=metrics,outputs=outputs,review_is_caller_declaration=True,source_resampled=False,alpha_changed=False,protected_rgb_exact=True,accepted=False,default_changed=False)
  (temp/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
  dest.mkdir();reserved=True
  try:
   for q in temp.iterdir():os.rename(q,dest/q.name)
  except BaseException:
   shutil.rmtree(dest);reserved=False;raise
  return manifest
 finally:
  shutil.rmtree(temp)

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 for n in NAMES:p.add_argument('--'+n,required=True)
 for n in ('spec','spec-sha256','output-dir'):p.add_argument('--'+n,required=True)
 return p
if __name__=='__main__':
 p=parser();a=p.parse_args()
 try:m=execute(**vars(a))
 except (ValueError,OSError) as e:p.exit(2,'REFUSED: '+str(e)+'\n')
 print(json.dumps(m,indent=2,allow_nan=False))
