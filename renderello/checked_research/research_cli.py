"""Reproducible source-bound research commands. Never promotes output to accepted quality.
Example: python -m renderello.checked_research.research_cli lighting --help
Hashes bind exact bytes, not semantic identity or alignment. --registration-reviewed is
an explicit caller declaration and is retained as such, never independent verification.
"""
import argparse, hashlib, json, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .evidence_checks import image_record, source_bound_alpha
from .detail_transfer import transfer
from .correspondence_render import render_correspondence

def file_record(path, expected):
 p=Path(path);digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if digest!=expected:raise ValueError('file hash mismatch: '+str(p))
 return dict(path=str(p.resolve()),sha256=digest,bytes=p.stat().st_size)

def execute(args):
 dest=Path(args.output_dir)
 if dest.exists():raise ValueError('output directory already exists; preserve prior evidence')
 if args.command=='lighting':
  binding=source_bound_alpha(args.source,args.alpha,args.source_sha256,expected_alpha_sha256=args.alpha_sha256)
  rawrec=image_record(args.raw,args.raw_sha256)
  with Image.open(args.source) as im:src=np.array(im.convert('RGB'))
  with Image.open(args.raw) as im:raw=np.array(im.convert('RGB'))
  with Image.open(args.alpha) as im:alpha=np.array(im)
  rgb,gain,metrics=transfer(src,raw,alpha,registration_reviewed=args.registration_reviewed)
  inputs=dict(binding=binding,raw=rawrec);extras={'gain.npy':gain};rgba=None
 else:
  inputs=dict(background=image_record(args.background,args.background_sha256),correspondence=file_record(args.correspondence,args.correspondence_sha256),valid_mask=file_record(args.valid,args.valid_sha256))
  with Image.open(args.background) as im:bg=np.array(im.convert('RGB'))
  with Image.open(args.valid) as im:
   if im.mode!='L':raise ValueError('explicit luminance valid mask required')
   valid=np.array(im)>127
  corr=np.load(args.correspondence,allow_pickle=False)
  rgb,alpha,metrics=render_correspondence(bg,corr,valid,out_of_image=args.out_of_image)
  rgba=np.dstack([rgb,alpha]);extras={}
 # All validation/calculation precedes creating output. Existing evidence never replaced.
 dest.parent.mkdir(parents=True,exist_ok=True)
 temp=Path(tempfile.mkdtemp(prefix='renderello-research-',dir=dest.parent))
 try:
  Image.fromarray(rgb).save(temp/'output.png')
  if rgba is not None:Image.fromarray(rgba,'RGBA').save(temp/'component-rgba.png')
  for name,arr in extras.items():np.save(temp/name,arr,allow_pickle=False)
  outputs={p.name:file_record(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in temp.iterdir()}
  for name,record in outputs.items():record['path']=str((dest/name).resolve())
  manifest=dict(command=args.command,inputs=inputs,metrics=metrics,outputs=outputs,registration_review_is_caller_declaration=bool(getattr(args,'registration_reviewed',False)),publication_accepted=False,quality_accuracy=None)
  (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  # mkdir is an exclusive reservation, so competing/existing output cannot be overwritten.
  dest.mkdir()
  try:
   for p in temp.iterdir():p.rename(dest/p.name)
  except BaseException:
   shutil.rmtree(dest);raise
  return manifest
 finally:shutil.rmtree(temp)

def parser():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 l=sub.add_parser('lighting');f=sub.add_parser('correspondence')
 for name in ('source','raw','alpha'):
  l.add_argument('--'+name,required=True);l.add_argument('--'+name+'-sha256',required=True)
 l.add_argument('--registration-reviewed',action='store_true')
 for name in ('background','correspondence','valid'):
  f.add_argument('--'+name,required=True);f.add_argument('--'+name+'-sha256',required=True)
 f.add_argument('--out-of-image',choices=('reject','mask','border'),default='reject')
 for s in (l,f):s.add_argument('--output-dir',required=True)
 return p
if __name__=='__main__':
 p=parser();a=p.parse_args()
 try:m=execute(a)
 except (ValueError,OSError) as e:p.exit(2,'REFUSED: '+str(e)+'\n')
 print(json.dumps(m,indent=2))
