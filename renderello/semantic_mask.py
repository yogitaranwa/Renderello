"""Explicit CPU IS-Net trial. Not automatic backend selection or semantic acceptance."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from PIL import Image

MODEL='isnet-general-use'
MODEL_URL='https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx'
MODEL_MD5='fc16ebd8b0c10d971d3513d564d01e29'


def cache_dir():
    # Models stay inside Renderello. Do not inherit a global U2NET_HOME.
    return Path(__file__).resolve().parent.parent/'.semantic-models'


def infer_file(source:Path,out:Path,allow_download=False):
    if out.exists():raise ValueError('output exists; preserve earlier attempts')
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds4million pixels; no silent resize')
        rgb=im.convert('RGB');pixels=np.array(rgb)
    cache=cache_dir();model=cache/(MODEL+'.onnx')
    if model.exists():
        if hashlib.md5(model.read_bytes()).hexdigest()!=MODEL_MD5:raise ValueError('cached model checksum mismatch; no automatic overwrite')
    elif not allow_download:raise ValueError('model not cached; --allow-download is required for the official free179MB model')
    try:
        from rembg import new_session
    except ImportError as e:raise RuntimeError('install optional requirements-semantic.txt in an isolated Renderello environment') from e
    os.environ['U2NET_HOME']=str(cache);os.environ.pop('MODEL_CHECKSUM_DISABLED',None)
    started=time.monotonic();session=new_session(MODEL,providers=['CPUExecutionProvider'])
    if hashlib.md5(model.read_bytes()).hexdigest()!=MODEL_MD5:raise ValueError('downloaded model checksum mismatch')
    matte=np.array(session.predict(rgb)[0])
    if matte.dtype!=np.uint8 or matte.shape!=pixels.shape[:2] or not np.any(matte):raise ValueError('invalid or empty inferred alpha')
    out.mkdir(parents=True)
    for name,data in [('source.png',pixels),('estimated-alpha.png',matte),('cutout.png',np.dstack([pixels,matte]))]:Image.fromarray(data).save(out/name)
    for label,color in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',rgb.size,color);bg.paste(rgb,mask=Image.fromarray(matte));bg.save(out/(label+'-review.png'))
    import importlib.metadata
    report=dict(model=MODEL,model_url=MODEL_URL,model_md5=MODEL_MD5,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                package='rembg',package_version=importlib.metadata.version('rembg'),provider='CPUExecutionProvider',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_rgb_unchanged=True,semantic_completeness='NOT_VERIFIED',source_framing='NOT_VERIFIED',quality='NOT_EVALUATED',seconds=time.monotonic()-started,
                warnings=['Inspect every part and hole. This mask failed the yellow notebook test.','A cropped source cannot become complete. Source RGB may still contain reflected background colors.','Optional model weights are the rembg-maintained ONNX conversion, not a claim of original author conversion.'],
                sources=['https://github.com/danielgatis/rembg','https://github.com/xuebinqin/DIS'])
    report['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'semantic-mask-manifest.json').write_text(json.dumps(report,indent=2));return report


def guarded_file(source,out,allow_download=False):
    """Per-image subprocess;180s wall-time. RSS check Linux-only,2.5GiB."""
    import subprocess,sys,tempfile
    cmd=[sys.executable,'-m','renderello.semantic_mask','--source',str(source),'--out',str(out)]
    if allow_download:cmd+=['--allow-download']
    env=os.environ.copy();env['OMP_NUM_THREADS']='2'
    with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
        p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,text=True,env=env);start=time.monotonic()
        try:
            while p.poll() is None:
                if time.monotonic()-start>180:raise RuntimeError('semantic-mask180second timeout; attempt not accepted')
                try:rss=int(next(l.split()[1] for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))
                except (FileNotFoundError,StopIteration):rss=0
                if rss>2560*1024:raise RuntimeError('semantic-mask2.5GiB RSS limit; attempt not accepted')
                time.sleep(.08)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        stdout.seek(0);stderr.seek(0)
        if p.returncode:raise RuntimeError('semantic-mask worker failed: '+stderr.read()[-2000:])
        return json.load(stdout)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-download',action='store_true');a=p.parse_args();print(json.dumps(infer_file(a.source,a.out,a.allow_download)))

if __name__=='__main__':main()
