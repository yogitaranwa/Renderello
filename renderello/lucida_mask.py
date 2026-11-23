"""Opt-in Lucida alpha research. Not a refractive model or accepted glass result."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image
REVISION='6cbedc9722652dc9a3df91dd871f0c4f3334e922'
MODEL='egeorcun/lucida'
# Pin publisher files, not a moving main revision or unrestricted remote code.
FILES={
 '__init__.py':('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',0),
 'BiRefNet_config.py':('e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52',298),
 'model.safetensors':('2f1aa6913426537d4b93dd5f7138ae5c6664e99abda98c95f3d5b9101283e7d5',884878856),
 'birefnet.py':('2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831',92134),
}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify_model(directory):
 directory=Path(directory)
 for name,(sha,size) in FILES.items():
  p=directory/name
  if p.is_symlink() or not p.is_file():raise ValueError('missing or symbolic publisher file: '+name)
  if p.stat().st_size!=size or digest(p)!=sha:raise ValueError('publisher file checksum mismatch: '+name)
 return directory

def export_review(source,alpha,out,*,inference=None):
 """Write RGB-preserving review artifacts; inferred alpha alone never earns acceptance."""
 source,out=Path(source),Path(out)
 if out.exists():raise ValueError('output exists; preserve earlier attempts')
 with Image.open(source) as im:
  if im.width*im.height>4_000_000:raise ValueError('source exceeds four million pixels')
  rgb=np.array(im.convert('RGB'))
 alpha=np.asarray(alpha)
 if alpha.dtype!=np.uint8 or alpha.shape!=rgb.shape[:2]:raise ValueError('alpha must be source-sized uint8')
 if not alpha.any():raise ValueError('empty inferred alpha')
 out.mkdir(parents=True)
 Image.fromarray(rgb).save(out/'source.png');Image.fromarray(alpha).save(out/'estimated-alpha.png')
 rgba=Image.fromarray(np.dstack([rgb,alpha]));rgba.save(out/'cutout.png')
 for title,color in [('white',(255,255,255)),('black',(0,0,0)),('magenta',(255,0,255))]:
  bg=Image.new('RGBA',rgba.size,(*color,255));Image.alpha_composite(bg,rgba).convert('RGB').save(out/(title+'-review.png'))
 report=dict(model=MODEL,revision=REVISION,source_sha256=digest(source),source_rgb_unchanged=True,
  alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),quality='NOT_EVALUATED',semantic_completeness='NOT_VERIFIED',
  source_framing='NOT_VERIFIED',refraction_estimated=False,background_color_removed=False,
  license_note='Publisher declares MIT weights/code. Training dataset rights are not independently cleared.',
  warnings=['Soft alpha is not refractive flow or transmission.','Source RGB retains reflected/refracted old background.','Source crops remain incomplete.','Inspect actual pixels on every review background.'],
  inference=inference or {})
 report['output_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
 (out/'lucida-mask-manifest.json').write_text(json.dumps(report,indent=2))
 return report

def infer_file(source,out,model_dir):
 """Use pre-reviewed pinned local code and safetensors, never auto-download."""
 import sys,importlib.util
 source,out=Path(source),Path(out)
 if out.exists():raise ValueError('output exists; preserve earlier attempts')
 with Image.open(source) as im:
  if im.width*im.height>4_000_000:raise ValueError('source exceeds four million pixels')
  rgb=im.convert('RGB')
 root=verify_model(model_dir)
 import torch,torchvision
 from safetensors.torch import load_file
 if not torch.cuda.is_available():raise RuntimeError('isolated CUDA trial only; no implicit CPU fallback')
 started=time.monotonic()
 spec=importlib.util.spec_from_file_location('_renderello_pinned_lucida',root/'__init__.py',submodule_search_locations=[str(root)])
 package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
 from importlib import import_module
 Model=import_module(spec.name+'.birefnet').BiRefNet
 Config=import_module(spec.name+'.BiRefNet_config').BiRefNetConfig
 model=Model(config=Config(bb_pretrained=False))
 model.load_state_dict(load_file(str(root/'model.safetensors'),device='cpu'),strict=True)
 model.eval().to('cuda')
 transform=torchvision.transforms.Compose([torchvision.transforms.Resize((1024,1024)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 with torch.inference_mode():prediction=model(transform(rgb).unsqueeze(0).to('cuda'))[-1].sigmoid().cpu()[0,0]
 a=Image.fromarray((prediction.numpy()*255).astype(np.uint8)).resize(rgb.size,Image.Resampling.BILINEAR)
 return export_review(source,np.array(a),out,inference=dict(device=torch.cuda.get_device_name(0),seconds=time.monotonic()-started,weight_sha256=FILES['model.safetensors'][0]))

def guarded_file(source,out,model_dir):
 import subprocess,sys
 cmd=[sys.executable,'-m','renderello.lucida_mask','--source',str(source),'--out',str(out),'--model-dir',str(model_dir)]
 try:p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 except subprocess.TimeoutExpired as e:raise RuntimeError('Lucida 180 second worker timeout; partial output is not accepted') from e
 if p.returncode:raise RuntimeError('Lucida worker failed: '+p.stderr[-2000:])
 return json.loads(p.stdout)

def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['source','out','model-dir']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();print(json.dumps(infer_file(a.source,a.out,a.model_dir)))
if __name__=='__main__':main()
