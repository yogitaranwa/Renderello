"""Explicit InSPyReNet fast CPU experiment. No automatic acceptance or download."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from PIL import Image

MODEL_URL='https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth'
MODEL_SHA256='e61ed7e85763dd83a796e1d650ea78604fff524121750858c54a6b560ac58c82'
PACKAGE_VERSION='1.3.4'


def _predict(rgb,checkpoint):
    import tempfile,contextlib,sys,importlib.metadata
    if importlib.metadata.version('transparent-background')!=PACKAGE_VERSION:
        raise RuntimeError('requires transparent-background=='+PACKAGE_VERSION)
    os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
    import torch
    from transparent_background import Remover
    torch.set_num_threads(2)
    # Keep package config inside this project, with a fresh known publisher config.
    # Explicit checkpoint avoids the package automatic downloader. Map output avoids
    # its default foreground recoloring; our saved RGB always comes from the source.
    with tempfile.TemporaryDirectory(prefix='.inspyrenet-',dir=Path(__file__).resolve().parent.parent) as td:
        old=os.environ.get('TRANSPARENT_BACKGROUND_FILE_PATH')
        os.environ['TRANSPARENT_BACKGROUND_FILE_PATH']=td
        try:
            with contextlib.redirect_stdout(sys.stderr):
                model=Remover(mode='fast',device='cpu',ckpt=str(checkpoint.resolve()),resize='static',jit=False)
                return np.array(model.process(rgb,type='map'))[:,:,0]
        finally:
            if old is None:os.environ.pop('TRANSPARENT_BACKGROUND_FILE_PATH',None)
            else:os.environ['TRANSPARENT_BACKGROUND_FILE_PATH']=old


def infer_file(source:Path,out:Path,checkpoint:Path):
    if out.exists():raise ValueError('output exists; preserve earlier attempts')
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds 4 million pixels; no silent source resize')
        if min(im.size)<3:raise ValueError('source too small')
        rgb=im.convert('RGB');pixels=np.array(rgb)
    if not checkpoint.is_file():raise ValueError('checkpoint missing; explicitly download the official free file first')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('checkpoint SHA256 mismatch; only tested fast model accepted')
    started=time.monotonic()
    try:alpha=_predict(rgb,checkpoint)
    except ImportError as e:raise RuntimeError('install optional requirements-pyramid.txt in an isolated Renderello environment') from e
    if alpha.dtype!=np.uint8 or alpha.shape!=pixels.shape[:2] or not np.any(alpha):raise ValueError('invalid or empty inferred alpha')
    out.mkdir(parents=True)
    for name,data in [('source.png',pixels),('estimated-alpha.png',alpha),('cutout.png',np.dstack([pixels,alpha]))]:Image.fromarray(data).save(out/name)
    for label,color in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',rgb.size,color);bg.paste(rgb,mask=Image.fromarray(alpha));bg.save(out/(label+'-review.png'))
    report=dict(model='InSPyReNet SwinB',mode='fast',input_preprocessing='static 384x384 internal model scale; alpha interpolated to unchanged source size',
        model_url=MODEL_URL,model_sha256=MODEL_SHA256,package='transparent-background',package_version=PACKAGE_VERSION,device='cpu',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_rgb_unchanged=True,semantic_completeness='NOT_VERIFIED',source_framing='NOT_VERIFIED',quality='NOT_EVALUATED',
        seconds=time.monotonic()-started,alpha_kind='saliency map, not optical matting',
        warnings=['Fast model loses the yellow notebook cover. Do not treat this as a general low-contrast solution.',
          'Silver scissors holes opened automatically, but edge color/softness and fine neck details still need review.',
          'Base 1024 mode was killed without an output in this environment. This command supports only the tested fast model.',
          'Original RGB is preserved, including source reflections and background color fringes. Cropped parts cannot be recovered.'],
        sources=['https://github.com/plemeri/InSPyReNet','https://github.com/plemeri/transparent-background','https://arxiv.org/abs/2209.09475'])
    report['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'pyramid-mask-manifest.json').write_text(json.dumps(report,indent=2));return report


def guarded_file(source,out,checkpoint):
    """CPU worker, 180 second timeout and Linux-only 2.5 GiB RSS guard."""
    import subprocess,sys,tempfile
    cmd=[sys.executable,'-m','renderello.pyramid_mask','--source',str(source),'--out',str(out),'--checkpoint',str(checkpoint)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
    with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
        p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,text=True,env=env);start=time.monotonic()
        try:
            while p.poll() is None:
                if time.monotonic()-start>180:raise RuntimeError('pyramid-mask 180 second timeout; attempt not accepted')
                try:rss=int(next(l.split()[1] for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))
                except (FileNotFoundError,StopIteration):rss=0
                if rss>2560*1024:raise RuntimeError('pyramid-mask 2.5 GiB RSS limit; attempt not accepted')
                time.sleep(.08)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        stdout.seek(0);stderr.seek(0)
        if p.returncode:raise RuntimeError('pyramid-mask worker failed (exit '+str(p.returncode)+'): '+stderr.read()[-2000:])
        return json.load(stdout)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(infer_file(a.source,a.out,a.checkpoint)))
if __name__=='__main__':main()
