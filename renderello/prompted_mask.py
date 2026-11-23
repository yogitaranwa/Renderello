"""Reviewed box/point Light HQ-SAM masks. Never automatic product acceptance."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image

MODEL_SHA256='0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196'
MODEL_REV='09b02a333b37772133eff3997bdba997867374b7'
MODEL_URL='https://huggingface.co/lkeab/hq-sam/resolve/'+MODEL_REV+'/sam_hq_vit_tiny.pth'


def validate_prompts(data,source,width,height):
    if not isinstance(data,dict) or data.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('prompts must bind to exact source SHA256')
    if data.get('image_size')!=[width,height]:raise ValueError('prompt image_size must match source width,height')
    if data.get('origin')!='reviewed_source_geometry':raise ValueError('prompt origin must explicitly declare reviewed_source_geometry; do not label automatic prompts reviewed')
    objects=data.get('objects')
    if not isinstance(objects,list) or not 1<=len(objects)<=8:raise ValueError('one to eight explicitly selected objects required')
    names=set()
    for item in objects:
        if not isinstance(item,dict) or not isinstance(item.get('name'),str) or not item['name'].strip() or len(item['name'])>80 or item['name'] in names:raise ValueError('each object needs a unique short name')
        names.add(item['name']);box=np.asarray(item.get('box'),dtype=float)
        if box.shape!=(4,) or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('invalid source-pixel box')
        points=item.get('points',[]);labels=item.get('labels',[])
        if not isinstance(points,list) or len(points)>32 or not isinstance(labels,list) or len(points)!=len(labels) or any(type(l)!=int or l not in (0,1) for l in labels):raise ValueError('points need equal0/1labels, max32')
        if points:
            p=np.asarray(points,dtype=float)
            if p.shape!=(len(points),2) or not np.isfinite(p).all() or np.any(p<0) or np.any(p[:,0]>=width) or np.any(p[:,1]>=height):raise ValueError('points must be finite source-pixel coordinates')
    return objects


def infer_file(source,out,prompts,checkpoint):
    if out.exists():raise ValueError('output exists; preserve attempts')
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds4million pixels; no silent resize')
        image=np.array(im.convert('RGB'));W,H=im.size
    data=json.loads(prompts.read_text());objects=validate_prompts(data,source,W,H)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('checkpoint does not match official pinned Light HQ-SAM SHA256')
    import torch
    from segment_anything_hq import sam_model_registry,SamPredictor
    torch.set_num_threads(2);started=time.monotonic();model=sam_model_registry['vit_tiny'](checkpoint=None)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval();predictor=SamPredictor(model)
    with torch.inference_mode():predictor.set_image(image)
    masks=[];scores=[]
    for item in objects:
        points=item.get('points',[])
        with torch.inference_mode():m,s,_=predictor.predict(box=np.asarray(item['box']),point_coords=np.asarray(points) if points else None,point_labels=np.asarray(item.get('labels')) if points else None,multimask_output=False)
        if m.shape!=(1,H,W) or not np.any(m):raise ValueError('empty/invalid prompted mask')
        masks.append(m[0].astype(bool));scores.append(s.tolist())
    alpha=np.logical_or.reduce(masks).astype('uint8')*255
    out.mkdir(parents=True)
    for i,a in enumerate(masks):Image.fromarray(a.astype('uint8')*255).save(out/f'object-{i+1}-alpha.png')
    for name,a in [('source.png',image),('estimated-alpha.png',alpha),('cutout.png',np.dstack([image,alpha]))]:Image.fromarray(a).save(out/name)
    for name,col in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',(W,H),col);bg.paste(Image.fromarray(image),mask=Image.fromarray(alpha));bg.save(out/(name+'-review.png'))
    (out/'reviewed-prompts.json').write_text(json.dumps(data,indent=2))
    report=dict(source_sha256=data['source_sha256'],prompt_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest(),prompt_origin=data['origin'],
                objects=[i['name'] for i in objects],union_rule='explicit object masks OR, not automatic semantic grouping',model='Light HQ-SAM vit_tiny',model_revision=MODEL_REV,model_url=MODEL_URL,model_sha256=MODEL_SHA256,
                source_rgb_unchanged=True,device='cpu',scores_unvalidated=scores,quality='NOT_EVALUATED',semantic_completeness='NOT_VERIFIED',seconds=time.monotonic()-started,
                warnings=['Reviewed prompts are manual input, not an automatic mask success.','High predicted scores do not prove correct holes, thin parts or semantic completeness.','Binary mask is not optical alpha. Background reflections and hard edge aliasing can remain.','Union includes only explicitly selected objects; omitted accessories remain omitted.'])
    report['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()};(out/'prompted-mask-manifest.json').write_text(json.dumps(report,indent=2));return report


def guarded_file(source,out,prompts,checkpoint):
    import subprocess,sys,tempfile,os
    cmd=[sys.executable,'-m','renderello.prompted_mask','--source',str(source),'--out',str(out),'--prompts',str(prompts),'--checkpoint',str(checkpoint)]
    env=os.environ.copy();env['OMP_NUM_THREADS']='2'
    with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
        p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,text=True,env=env);start=time.monotonic()
        try:
            while p.poll() is None:
                if time.monotonic()-start>180:raise RuntimeError('prompted-mask180second timeout; not accepted')
                try:rss=int(next(l.split()[1] for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))
                except (FileNotFoundError,StopIteration):rss=0
                if rss>2560*1024:raise RuntimeError('prompted-mask2.5GiB RSS limit; not accepted')
                time.sleep(.08)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        stdout.seek(0);stderr.seek(0)
        if p.returncode:raise RuntimeError('prompted-mask worker failed: '+stderr.read()[-2000:])
        return json.load(stdout)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','out','prompts','checkpoint'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(infer_file(a.source,a.out,a.prompts,a.checkpoint)))

if __name__=='__main__':main()
