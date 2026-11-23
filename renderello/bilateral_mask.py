"""Explicit pinned BiRefNet-lite ONNX CPU experiment. No prompts/downloads."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image

MODEL_URL='https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx'
MODEL_SHA256='5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333'
ORT_VERSION='1.31.1'


def _predict(rgb,checkpoint):
    import importlib.metadata
    if importlib.metadata.version('onnxruntime')!=ORT_VERSION:raise RuntimeError('requires onnxruntime=='+ORT_VERSION)
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.enable_cpu_mem_arena=False;opts.enable_mem_pattern=False
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(checkpoint.resolve()),sess_options=opts,providers=['CPUExecutionProvider'])
    # Match rembg conversion normalization, including max-channel normalization.
    x=np.array(rgb.resize((1024,1024),Image.Resampling.LANCZOS)).astype(np.float32)
    x/=max(float(x.max()),1e-6)
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    raw=session.run(None,{session.get_inputs()[0].name:x.transpose(2,0,1)[None]})[0][:,0,:,:]
    return normalize_logits(raw,rgb.size)


def normalize_logits(raw,size):
    if raw.shape!=(1,1024,1024) or not np.isfinite(raw).all():raise ValueError('invalid logits shape or nonfinite values')
    pred=1/(1+np.exp(-np.clip(raw,-80,80)))
    span=float(pred.max()-pred.min())
    if span<1e-8:raise ValueError('constant prediction; no valid foreground')
    pred=(pred-pred.min())/span
    return np.array(Image.fromarray((pred.squeeze()*255).astype(np.uint8)).resize(size,Image.Resampling.LANCZOS))


def infer_file(source:Path,out:Path,checkpoint:Path):
    if out.exists():raise ValueError('output exists; preserve earlier attempts')
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds 4 million pixels; no silent source resize')
        if min(im.size)<3:raise ValueError('source too small')
        rgb=im.convert('RGB');pixels=np.array(rgb)
    if not checkpoint.is_file():raise ValueError('checkpoint missing; explicitly download the official free conversion first')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('checkpoint SHA256 mismatch')
    started=time.monotonic()
    try:alpha=_predict(rgb,checkpoint)
    except ImportError as e:raise RuntimeError('install optional requirements-bilateral.txt in an isolated Renderello environment') from e
    if alpha.dtype!=np.uint8 or alpha.shape!=pixels.shape[:2] or not np.any(alpha):raise ValueError('invalid or empty inferred alpha')
    out.mkdir(parents=True)
    for name,data in [('source.png',pixels),('estimated-alpha.png',alpha),('cutout.png',np.dstack([pixels,alpha]))]:Image.fromarray(data).save(out/name)
    for label,color in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',rgb.size,color);bg.paste(rgb,mask=Image.fromarray(alpha));bg.save(out/(label+'-review.png'))
    report=dict(model='BiRefNet general-lite',conversion='rembg-maintained ONNX, not original author checkpoint bytes',model_url=MODEL_URL,model_sha256=MODEL_SHA256,
        runtime='onnxruntime',runtime_version=ORT_VERSION,provider='CPUExecutionProvider',internal_input=[1024,1024],graph_optimizations=False,cpu_mem_arena=False,mem_pattern=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_rgb_unchanged=True,semantic_completeness='NOT_VERIFIED',source_framing='NOT_VERIFIED',quality='NOT_EVALUATED',seconds=time.monotonic()-started,
        alpha_kind='dichotomous segmentation, not optical glass matting',
        warnings=['Masks can omit wanted objects, include unwanted props, or bridge foreground and backdrop. Review each source separately.',
          'Source RGB is preserved, including source-color fringes and reflections. Cropped parts cannot be recovered.',
          'This model was killed in the smaller research environment. Successful CPU tests used Kaggle. Owner-PC resources and execution unverified.'],
        sources=['https://github.com/ZhengPeng7/BiRefNet','https://huggingface.co/ZhengPeng7/BiRefNet_lite','https://github.com/danielgatis/rembg'])
    report['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'bilateral-mask-manifest.json').write_text(json.dumps(report,indent=2));return report


def guarded_file(source,out,checkpoint):
    """Separate CPU worker, 180s timeout and Linux-only 8GiB RSS guard."""
    import subprocess,sys,tempfile,os
    # Standalone module script keeps core CLI imports out of the inference worker.
    cmd=[sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--out',str(out),'--checkpoint',str(checkpoint)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
        p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,text=True,env=env);start=time.monotonic()
        try:
            while p.poll() is None:
                if time.monotonic()-start>180:raise RuntimeError('bilateral-mask 180 second timeout; attempt not accepted')
                try:rss=int(next(l.split()[1] for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))
                except (FileNotFoundError,StopIteration):rss=0
                if rss>8192*1024:raise RuntimeError('bilateral-mask 8GiB RSS limit; attempt not accepted')
                time.sleep(.1)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        stdout.seek(0);stderr.seek(0)
        if p.returncode:raise RuntimeError('bilateral-mask worker failed (exit '+str(p.returncode)+'): '+stderr.read()[-2000:])
        return json.load(stdout)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(infer_file(a.source,a.out,a.checkpoint)))
if __name__=='__main__':main()
