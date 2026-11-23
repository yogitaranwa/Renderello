"""Readiness runner for Renderello v0.4.58. Default is plan-only.
GPU mode is UNEXECUTED and requires separately verified owner/runtime authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

VERSION = '0.4.58'
CASES = [('shampoo2','flat_lay'),('snack1','flat_lay'),('watch_full','flat_lay')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def code_hash(root):
    h = hashlib.sha256()
    for p in sorted((root/'renderello').rglob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    registry=root/'renderello/source_registry.json'
    h.update(registry.read_bytes())
    return h.hexdigest()


def plan(root):
    rows=[]
    p=root/'benchmark/sources/watch_full.jpg'
    for material in ['linen','slate','paper']:
        rows.append({'key':'watch_full_flat_lay_'+material,'source':str(p.resolve()),
                     'source_sha256':sha(p),'template':'flat_lay','seed':0,'material':material})
    return {'version_required':VERSION,'code_sha256':code_hash(root),'cases':rows,
            'scope':'Same fullwatch source andseed,3explicit surface materials. Not20source acceptance.',
            'gpu_status':'NOT_EXECUTED','authority':'Readiness is not permission to start a runtime.'}


def validate_checkpoint(path, signature):
    if not path.is_file():return False
    m=json.loads(path.read_text())
    if m.get('signature')!=signature:raise ValueError('checkpoint signature mismatch')
    root=path.parent.resolve()
    if not m.get('artifacts'):raise ValueError('checkpoint has no artifacts')
    for rel,digest in m['artifacts'].items():
        p=(root/rel).resolve()
        if root not in p.parents or not p.is_file() or sha(p)!=digest:
            raise ValueError('checkpoint artifact missing/changed/unsafe: '+rel)
    if m.get('status')!='CAPTURED_REVIEW_REQUIRED':
        raise ValueError('checkpoint records failure, do not silently retry: '+str(path))
    return True


def worker(args):
    # Heavy imports happen only in this isolated child, never plan/resume checks.
    sys.path.insert(0,str(args.code_root.resolve()))
    import renderello
    if renderello.__version__!=VERSION:raise RuntimeError('wrong Renderello version')
    from renderello.pipeline import run_pipeline
    from renderello.config import RunConfig,DeviceConfig
    from renderello.schemas import EditRequest,Route
    from renderello.templates import get_template
    from renderello.iclight_gpu import ICLight,FlatlaySurfaceCallback
    from PIL import Image
    import numpy as np
    case=json.loads(args.case_file.read_text())
    if sha(case['source'])!=case['source_sha256']:raise RuntimeError('source bytes changed')
    ic=None
    mode=args.mode
    try:
        if mode=='gpu':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback')
            if not args.weights or not args.weights.is_file():raise RuntimeError('provide verified local IC-Light weights before execution')
            if sha(args.weights)!=args.weights_sha256:raise RuntimeError('IC-Light weight hash mismatch')
            ic=ICLight(str(args.weights))
            scene=FlatlaySurfaceCallback(ic,seed=case['seed'],material=case['material'])
            device=DeviceConfig.resolve(prefer_gpu=True)
        else:
            # Plumbing rehearsal only. Uses existing GrabCut + flat mock scene.
            scene=lambda gray,prompt:np.full_like(gray,(218,236,223))
            device=DeviceConfig()
        mask=None;glass=case['key'].startswith('glass_full_')
        if glass:
            from renderello.compose import estimate_clean_backdrop,silhouette_from_backdrop
            rgb=np.array(Image.open(case['source']).convert('RGB'));B,ok=estimate_clean_backdrop(rgb)
            if not ok:raise RuntimeError('glass source backdrop not clean')
            mask=args.case_dir/'glass-clean-backdrop-alpha.png'
            Image.fromarray((silhouette_from_backdrop(rgb,B)*255).astype('uint8')).save(mask)
        result=run_pipeline(EditRequest(Path(case['source']),get_template(case['template']),seed=case['seed'],mask_path=mask),
                            [Route.SCENE_COMPOSE],RunConfig(seed=case['seed'],variants=1,
                            out_root=args.case_dir/'runs',device=device),scene_fn=scene,scene_finish=True,glass_mode=glass)
        m=json.loads(result.manifest_path.read_text())
        if mode=='gpu' and m.get('cutout',{}).get('backend')!=('supplied_mask' if glass else 'birefnet'):
            raise RuntimeError('GPU capture rejected non-BiRefNet fallback')
        if not result.variants or not result.variants[0].output_path.is_file():
            raise RuntimeError('pipeline produced no final output; inspect run manifest')
        alpha=Path(m['cutout']['matte_path'])
        if not alpha.is_file():raise RuntimeError('source alpha missing')
        layer_dir=result.run_dir/'intermediates/scene_layers_seed0'
        for name in ['scene.png','product_rgba.png','layers.json']:
            if not (layer_dir/name).is_file():raise RuntimeError('required saved layer missing: '+name)
        Image.open(alpha).verify()
        record={'mode':'GPU_UNVALIDATED_CAPTURE' if mode=='gpu' else 'MOCK_REHEARSAL_NOT_GPU',
                'run_manifest':str(result.manifest_path),'source_sha256':case['source_sha256'],
                'code_sha256':code_hash(args.code_root),'scene_finish':True,'glass_mode':glass,'mask_status':'engineer-selected clean-backdrop silhouette' if glass else 'BiRefNet', 'experiment':'flatlay_surface_materials','material':case.get('material'),
                'quality_status':'NOT_EVALUATED','warning':'Capture completion is not mask, scene, or human quality acceptance.'}
        atomic_json(args.case_dir/'capture-record.json',record)
        print(json.dumps(record),flush=True)
    finally:
        if ic is not None:ic.close()


def stop_group(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def run(args):
    args.code_root=args.code_root.resolve()
    args.out=args.out.resolve()
    if args.weights:args.weights=args.weights.resolve()
    p=plan(args.code_root)
    if not args.rehearse and not args.execute_gpu:
        print(json.dumps(p,indent=2));return 0
    if args.execute_gpu and (not args.weights or not args.weights_sha256):
        raise ValueError('GPU execution requires local IC-Light weights and independently verified SHA256')
    if not 1<=args.budget_seconds<=4800 or not 1<=args.case_timeout<=600:
        raise ValueError('budget must be1-4800s, case timeout1-600s')
    args.out.mkdir(parents=True,exist_ok=True)
    mode='gpu' if args.execute_gpu else 'mock'
    signature=sha(Path(__file__))+'|'+p['code_sha256']+'|'+mode+'|'+str(args.weights_sha256)
    lock=args.out/'plan.json'
    bound={'signature':signature,'mode':mode,'plan':p}
    if lock.exists() and json.loads(lock.read_text())!=bound:
        raise ValueError('output root belongs to different code/mode/plan; use fresh root')
    atomic_json(lock,bound)
    deadline=time.monotonic()+args.budget_seconds
    summary=[]
    for case in p['cases']:
        dst=args.out/case['key'];checkpoint=dst/'checkpoint.json'
        if validate_checkpoint(checkpoint,signature):
            summary.append({'key':case['key'],'status':'RESUME_VERIFIED_SKIP'});continue
        if dst.exists():raise ValueError('partial case directory exists; inspect and choose fresh root, no blind retry')
        remaining=deadline-time.monotonic()
        if remaining<=1:
            summary.append({'key':case['key'],'status':'BUDGET_EXHAUSTED_BEFORE_CASE'});break
        dst.mkdir();case_file=dst/'case.json';atomic_json(case_file,case)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--code-root',str(args.code_root),
             '--case-file',str(case_file),'--case-dir',str(dst),'--mode',mode]
        if args.weights:cmd+=['--weights',str(args.weights),'--weights-sha256',args.weights_sha256]
        with (dst/'worker.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=child.wait(timeout=min(args.case_timeout,remaining))
            except subprocess.TimeoutExpired:
                stop_group(child);rc=-1
            except BaseException:
                stop_group(child);raise
        artifacts={str(f.relative_to(dst)):sha(f) for f in sorted(dst.rglob('*')) if f.is_file()}
        status='CAPTURED_REVIEW_REQUIRED' if rc==0 else 'FAILED_OR_TIMED_OUT'
        atomic_json(checkpoint,{'signature':signature,'status':status,'exit':rc,'artifacts':artifacts,
                                'quality_status':'NOT_EVALUATED','mode':mode})
        summary.append({'key':case['key'],'status':status,'exit':rc})
        atomic_json(args.out/'summary.json',summary)
        if rc!=0:break
    atomic_json(args.out/'summary.json',summary)
    print(json.dumps({'mode':mode,'results':summary,'runtime_note':'STOP hosted session separately. This script cannot confirm provider runtime shutdown.'},indent=2))
    return 0 if all(s['status'] in ['CAPTURED_REVIEW_REQUIRED','RESUME_VERIFIED_SKIP'] for s in summary) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('capture-smoke'))
    g=p.add_mutually_exclusive_group();g.add_argument('--rehearse',action='store_true');g.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--budget-seconds',type=int,default=900);p.add_argument('--case-timeout',type=int,default=300)
    p.add_argument('--weights',type=Path);p.add_argument('--weights-sha256')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--case-file',type=Path,help=argparse.SUPPRESS);p.add_argument('--case-dir',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=['gpu','mock'],default='mock',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:worker(args);return 0
    return run(args)

if __name__=='__main__':raise SystemExit(main())
