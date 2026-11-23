"""Explicit experimental U2NETP masks. No download, default selection or quality pass.
Only accepts inspected original-publisher architecture and weights by SHA256.
Run this command with a CPU torch interpreter in its own environment.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

CODE_SHA256='96dd7a19c7de4f13520ccfc1075ded3350ff4946493be3561b9918a46218f415'
WEIGHTS_SHA256='e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854'
FULL_WEIGHTS_SHA256='10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58'
COMMIT='ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29'


def verified_file(path:Path,expected:str)->str:
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise ValueError(f'publisher artifact hash mismatch: {path.name}')
    return actual


def estimate(source:Path,code:Path,weights:Path,out:Path,model:str="u2netp"):
    if model not in ("u2netp","u2net"):raise ValueError("unsupported model")
    weight_hash = WEIGHTS_SHA256 if model == "u2netp" else FULL_WEIGHTS_SHA256
    verified_file(code,CODE_SHA256);verified_file(weights,weight_hash)
    # Verification must precede module import and tensor deserialization.
    import importlib.util
    import resource
    import numpy as np
    import torch
    from PIL import Image
    if out.exists():raise ValueError('output directory exists; use a fresh directory')
    out.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('_verified_u2net',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net=(module.U2NETP if model == "u2netp" else module.U2NET)(3,1)
    net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));net.eval()
    im=Image.open(source).convert('RGB')
    x=np.asarray(im.resize((320,320),Image.Resampling.BILINEAR)).astype('float32')
    x/=max(float(x.max()),1.)
    x=(x-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
    t=time.monotonic()
    with torch.inference_mode():raw=net(torch.from_numpy(x.transpose(2,0,1).copy())[None])[0][0,0].numpy()
    if not np.isfinite(raw).all():raise ValueError('non-finite model output')
    lo,hi=float(raw.min()),float(raw.max())
    if hi-lo<1e-8:raise ValueError('constant saliency; no reliable foreground range')
    alpha=Image.fromarray(((raw-lo)/(hi-lo)*255).astype('uint8')).resize(im.size,Image.Resampling.BILINEAR)
    np.save(out/'raw-saliency-320.npy',raw)
    alpha.save(out/'estimated-alpha.png')
    rgba=im.copy();rgba.putalpha(alpha);rgba.save(out/'estimated-cutout.png')
    preview=Image.new('RGB',im.size,(218,236,223));preview.paste(im,(0,0),alpha);preview.save(out/'review-preview.png')
    manifest={'status':'REVIEW_REQUIRED','backend':model+'_experimental','source':str(source.resolve()),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_commit':COMMIT,
              'code_sha256':CODE_SHA256,'weights_sha256':weight_hash,'input_size':[320,320],
              'torch_version':torch.__version__,'inference_seconds':time.monotonic()-t,
              'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'raw_probability_range':[lo,hi],'raw_saliency':'raw-saliency-320.npy',
              'alpha_sha256':hashlib.sha256((out/'estimated-alpha.png').read_bytes()).hexdigest(),
              'alpha':'estimated-alpha.png',
              'warning':'New estimated saliency, not recovered GPU alpha or quality acceptance. Inspect missing objects, halos, disconnected fragments and transparency. No component removal or edge cleanup applied.'}
    (out/'mask-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['u2netp','u2net'],default='u2netp',help='explicit original-publisher variant; both experimental')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    # The parent does not import torch; it can stop the child before OOM.
    if args.worker:
        print(json.dumps(estimate(args.source,args.code,args.weights,args.out,args.model)),flush=True);return 0
    verified_file(args.code,CODE_SHA256);verified_file(args.weights,WEIGHTS_SHA256 if args.model == "u2netp" else FULL_WEIGHTS_SHA256)
    if args.out.exists():p.error('output directory exists; use a fresh directory')
    child=subprocess.Popen([sys.executable,'-m','renderello.cpu_mask',*sys.argv[1:],'--worker'])
    start=time.monotonic();reason=None
    while child.poll() is None:
        try:
            text=Path(f'/proc/{child.pid}/status').read_text()
            rss=int(next(line.split()[1] for line in text.splitlines() if line.startswith('VmRSS:')))
        except (FileNotFoundError,StopIteration):rss=0
        if rss>1500*1024:reason='1500MiB RSS safety limit'
        if time.monotonic()-start>90:reason='90second safety timeout'
        if reason:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            print('stopped: '+reason,file=sys.stderr);return 2
        time.sleep(.08)
    return child.returncode

if __name__=='__main__':raise SystemExit(main())
