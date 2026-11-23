"""Opt-in trimap matting and foreground recovery, not automatic segmentation.

CF/LKM and multi-level foreground estimation: official MIT PyMatting 1.1.16.
Guided filter: He, Sun, Tang (ECCV2010), RGB local linear equations 14-16.
No network access. This stage cannot recover a missing semantic object or pose.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SOURCES=['https://github.com/pymatting/pymatting/',
         'https://pymatting.github.io/alpha.html',
         'https://pymatting.github.io/foreground.html',
         'https://arxiv.org/abs/2006.14970',
         'https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf']


def validate(rgb,alpha):
    if rgb.dtype!=np.uint8 or alpha.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or alpha.shape!=rgb.shape[:2]:
        raise ValueError('source RGB and source-sized alpha must be uint8')
    if alpha.size>1_000_000:raise ValueError('matting is capped at one million source pixels; no silent resize')
    if alpha.size<9:raise ValueError('source too small')


def make_trimap(alpha,radius=3):
    """Conservative heuristic seeds, NOT ground-truth confidence or semantic proof."""
    if alpha.dtype!=np.uint8 or alpha.ndim!=2:raise ValueError('alpha must be uint8 grayscale')
    if not isinstance(radius,int) or not 1<=radius<=12:raise ValueError('radius must be integer1-12')
    k=np.ones((2*radius+1,2*radius+1),np.uint8)
    fg=cv2.erode((alpha>240).astype(np.uint8),k,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    bg=cv2.erode((alpha<15).astype(np.uint8),k,borderType=cv2.BORDER_CONSTANT,borderValue=1)>0
    tri=np.full(alpha.shape,128,np.uint8);tri[fg]=255;tri[bg]=0
    check_trimap(tri,alpha.shape)
    return tri


def check_trimap(tri,shape):
    if tri.dtype!=np.uint8 or tri.shape!=shape or not set(np.unique(tri)).issubset({0,128,255}):
        raise ValueError('trimap must be source-sized uint8 with0,128,255')
    if not (np.any(tri==0) and np.any(tri==255)):
        raise ValueError('trimap needs definite background and foreground')


def guided_alpha(rgb,alpha,radius=4,eps=.001):
    """RGB guided filter; edge-aware feathering only, no foreground-color recovery."""
    I=rgb.astype(np.float64)/255;p=alpha.astype(np.float64)/255
    box=lambda a:cv2.boxFilter(a,-1,(2*radius+1,2*radius+1),normalize=True,borderType=cv2.BORDER_REFLECT)
    meanI=box(I);meanP=box(p);covIp=box(I*p[...,None])-meanI*meanP[...,None]
    cov=np.empty((*p.shape,3,3),np.float64)
    for i in range(3):
        for j in range(3):cov[...,i,j]=box(I[...,i]*I[...,j])-meanI[...,i]*meanI[...,j]+(eps if i==j else 0)
    coeff=np.linalg.solve(cov,covIp[...,None])[...,0];b=meanP-(coeff*meanI).sum(2)
    return np.clip((box(coeff)*I).sum(2)+box(b),0,1)


def refine_arrays(rgb,alpha,trimap,method='cf',edge_colors=False):
    validate(rgb,alpha);check_trimap(trimap,alpha.shape)
    if method not in ('cf','lkm','guided'):raise ValueError('unsupported matting method')
    if int((trimap==128).sum())>200_000:raise ValueError('unknown trimap band exceeds200000pixels; supply a tighter reviewed trimap')
    image=rgb.astype(np.float64)/255;tri=trimap.astype(np.float64)/255
    tri[trimap==128]=.5
    if method=='guided':a=guided_alpha(rgb,alpha)
    else:
        try:
            from pymatting import estimate_alpha_cf,estimate_alpha_lkm
        except ImportError as e:raise RuntimeError('optional PyMatting1.1.16 is not installed; install official matting requirements in an isolated project environment') from e
        a=(estimate_alpha_cf if method=='cf' else estimate_alpha_lkm)(image,tri)
    if not np.isfinite(a).all() or a.shape!=alpha.shape:raise ValueError('invalid estimated alpha')
    a=np.clip(a,0,1);a[trimap==0]=0;a[trimap==255]=1
    result=rgb.copy()
    if edge_colors:
        try:from pymatting import estimate_foreground_ml
        except ImportError as e:raise RuntimeError('foreground recovery needs optional official PyMatting1.1.16') from e
        f=estimate_foreground_ml(image,a)
        if f.shape!=image.shape or not np.isfinite(f).all():raise ValueError('invalid estimated foreground')
        result=np.round(np.clip(f,0,1)*255).astype(np.uint8)
        # Never estimate the protected interior. Restore byte-for-byte.
        result[trimap==255]=rgb[trimap==255]
        result[trimap==0]=rgb[trimap==0]
    matte=np.round(a*255).astype(np.uint8)
    changed=np.any(result!=rgb,axis=2)
    return result,matte,dict(protected_core_pixels=int((trimap==255).sum()),
        protected_core_rgb_max_error=int(np.abs(result[trimap==255].astype('int16')-rgb[trimap==255].astype('int16')).max()),
        changed_rgb_pixels=int(changed.sum()),alpha_changed_pixels=int((matte!=alpha).sum()),
        source_rgb_unchanged=not edge_colors,entire_product_source_exact=False,edge_rgb_estimated=bool(edge_colors),
        semantic_completeness='NOT_VERIFIED',quality='NOT_EVALUATED')


def refine_file(source:Path,alpha:Path,out:Path,*,trimap_path=None,method='cf',radius=3,edge_colors=False):
    if out.exists():raise ValueError('output directory exists; preserve original attempts')
    with Image.open(source) as im:rgb=np.array(im.convert('RGB'))
    with Image.open(alpha) as im:
        if im.mode!='L':raise ValueError('alpha must be8-bit grayscale')
        a=np.array(im)
    validate(rgb,a)
    if trimap_path:
        with Image.open(trimap_path) as im:
            if im.mode!='L':raise ValueError('trimap must be8-bit grayscale')
            tri=np.array(im)
    else:tri=make_trimap(a,radius)
    check_trimap(tri,a.shape);t=time.monotonic();foreground,matte,meta=refine_arrays(rgb,a,tri,method,edge_colors)
    out.mkdir(parents=True)
    for name,data in [('source.png',rgb),('initial-alpha.png',a),('trimap.png',tri),('foreground.png',foreground),('refined-alpha.png',matte),('refined-cutout.png',np.dstack([foreground,matte])),('changed-rgb.png',(np.any(foreground!=rgb,2)*255).astype('uint8'))]:
        Image.fromarray(data).save(out/name)
    for name,color in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',(rgb.shape[1],rgb.shape[0]),color);bg.paste(Image.fromarray(foreground),mask=Image.fromarray(matte));bg.save(out/(name+'-review.png'))
    meta.update(method=method,backend='pymatting1.1.16' if method!='guided' or edge_colors else 'local_RGB_guided_filter',
        trimap_origin='supplied_reviewer_input' if trimap_path else 'heuristic_from_estimated_alpha_not_ground_truth',
        radius=radius,seconds=time.monotonic()-t,source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        warnings=['Opt-in research stage, not automatic quality acceptance.','Cannot repair missing objects, change camera pose, recover true glass optics or lighting.',
        'Heuristic trimap can lock wrong foreground/background; inspect every part and hole.',
        'Foreground recovery changes boundary RGB; only the saved definite-foreground core is restored exactly.'] if edge_colors else ['Alpha changed; source RGB unchanged. Camera/material/missing-object accuracy not verified.'],sources=SOURCES)
    meta['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'matting-manifest.json').write_text(json.dumps(meta,indent=2));return meta


MAX_WORKER_RSS_BYTES=1500*1024*1024
MAX_WORKER_SECONDS=90


def _stop_worker(child):
    import subprocess
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


def guarded_file(source,alpha,out,*,trimap_path=None,method='cf',radius=3,edge_colors=False):
    """Sampled90s/1500MiB single-worker guard. Not an OS allocation ceiling."""
    import subprocess,sys,os,tempfile
    from .process_memory import ResidentMemory
    # Fail before starting work if this host cannot measure its own resident memory.
    preflight=ResidentMemory(os.getpid())
    try:preflight.sample()
    finally:preflight.close()
    cmd=[sys.executable,'-m','renderello.matting','--source',str(source),'--alpha',str(alpha),'--out',str(out),'--method',method,'--radius',str(radius)]
    if trimap_path:cmd+=['--trimap',str(trimap_path)]
    if edge_colors:cmd+=['--estimate-edge-colors']
    # Files avoid a full stdout/stderr pipe blocking the child before it can exit.
    # Imported callers can run outside the repository. Child module discovery
    # must not depend on the caller's current working directory.
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
    with tempfile.TemporaryFile(mode='w+t') as stdout_file, tempfile.TemporaryFile(mode='w+t') as stderr_file:
        child=subprocess.Popen(cmd,stdout=stdout_file,stderr=stderr_file,text=True,env=env)
        monitor=None
        exit_reap_grace_used=False
        start=time.monotonic()
        try:
            if child.poll() is None:
                try:monitor=ResidentMemory(child.pid)
                except Exception as exc:
                    if child.poll() is None:raise RuntimeError('memory monitoring unavailable; attempt not accepted') from exc
            while child.poll() is None:
                if time.monotonic()-start>MAX_WORKER_SECONDS:raise RuntimeError('90second matting timeout; attempt not accepted')
                try:rss=monitor.sample()
                except Exception as exc:
                    if child.poll() is not None:break  # exited between poll and sample
                    # Linux may remove VmRSS while the process is tearing down,
                    # just before waitpid sees its exit. Reap once, bounded200ms.
                    # A still-running unmonitorable worker remains refused; a
                    # nonzero reaped exit is rejected below, never legitimized.
                    try:child.wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('memory monitoring unavailable; attempt not accepted') from exc
                    exit_reap_grace_used=True
                    break
                if rss>MAX_WORKER_RSS_BYTES:raise RuntimeError('1500MiB RSS safety limit; attempt not accepted')
                time.sleep(.08)
            if time.monotonic()-start>MAX_WORKER_SECONDS:raise RuntimeError('90second matting timeout; attempt not accepted')
            stdout_file.seek(0);stderr_file.seek(0)
            stdout,stderr=stdout_file.read(),stderr_file.read()
            if child.returncode:raise RuntimeError('matting worker failed: '+stderr[-2000:])
            result=json.loads(stdout)
            if exit_reap_grace_used:result['worker_guard_exit_reap_wait_seconds']=.2
            return result
        finally:
            try:_stop_worker(child)
            finally:
                if monitor is not None:monitor.close()


def main():
    import argparse,os
    os.environ.setdefault('NUMBA_NUM_THREADS','1')
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','alpha','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--trimap',type=Path);p.add_argument('--method',choices=['cf','lkm','guided'],default='cf');p.add_argument('--radius',type=int,default=3);p.add_argument('--estimate-edge-colors',action='store_true');a=p.parse_args()
    print(json.dumps(refine_file(a.source,a.alpha,a.out,trimap_path=a.trimap,method=a.method,radius=a.radius,edge_colors=a.estimate_edge_colors)))

if __name__=='__main__':main()


def compose_bundle(bundle:Path,background:Path,out:Path,*,source_pose,material,framing,fill=.68):
    """Explicit top-down opaque route with original-source and estimated-edge provenance."""
    from .compose import fit_product,flatlay_center,finish_scene
    from .cutout import _bbox_from_matte
    if out.exists():raise ValueError('output directory exists; preserve attempts')
    if source_pose!='overhead':raise ValueError('this route requires a reviewed overhead source, not inferred pose')
    if material!='opaque':raise ValueError('this route cannot paste transparent material')
    if framing!='complete':raise ValueError('this route requires a reviewed complete product; cropped parts cannot be recovered')
    if not .2<=fill<=.85:raise ValueError('fill must be0.2-0.85')
    meta=json.loads((bundle/'matting-manifest.json').read_text())
    for name in ('source.png','foreground.png','refined-alpha.png','refined-cutout.png','trimap.png'):
        digest=hashlib.sha256((bundle/name).read_bytes()).hexdigest()
        if digest!=meta['output_sha256'].get(name):raise ValueError('matting bundle hash mismatch: '+name)
    p=np.array(Image.open(bundle/'refined-cutout.png').convert('RGBA'))
    with Image.open(background) as im:scene=np.array(im.convert('RGB'))
    H,W=scene.shape[:2]
    if max(H,W)>2048 or min(H,W)<64:raise ValueError('background must be64-2048pixels per side')
    crop,xy=fit_product(p,_bbox_from_matte(p[...,3]),(W,H),fill);xy=flatlay_center(crop[...,3],(W,H));result,_=finish_scene(scene,crop,xy,shadow_style='surface contact')
    out.mkdir(parents=True);Image.fromarray(result).save(out/'composite.png')
    report=dict(original_source_sha256=meta['source_sha256'],matting_manifest_sha256=hashlib.sha256((bundle/'matting-manifest.json').read_bytes()).hexdigest(),
        background_sha256=hashlib.sha256(background.read_bytes()).hexdigest(),background_camera='reviewer_declared_overhead',
        source_pose=source_pose,material=material,framing=framing,placement=list(xy),
        source_rgb_unchanged=meta.get('source_rgb_unchanged',not meta['edge_rgb_estimated']),edge_rgb_estimated=meta['edge_rgb_estimated'],
        protected_core_rgb_max_error_before_placement=meta['protected_core_rgb_max_error'],
        warning='Source RGB may be resampled once for placement. Edge RGB estimation is not exact-source product fidelity.2D shadow is not recovered physical contact.',
        quality='NOT_EVALUATED',composite_sha256=hashlib.sha256((out/'composite.png').read_bytes()).hexdigest())
    (out/'compose-manifest.json').write_text(json.dumps(report,indent=2));return report


def refine_native_region(rgb, alpha, roi, *, source_sha256, alpha_sha256,
                         reviewed=False, review_note='', material=None,
                         trimap=None, method='cf', edge_colors=False, feather=8):
    """Native-resolution local matting for a larger source, without global solves.

    ROI and trimap are explicit reviewer hypotheses. Taper to the original prior
    at the rectangle border instead of pasting a hard solver seam. This cannot
    recover missing parts or prove semantic/optical correctness. No default route
    is changed, and no scene/camera/lighting or acceptance is inferred.
    """
    from .candidate_diagnostics import digest
    if type(edge_colors)is not bool:raise ValueError('explicit boolean edge-colour mode required')
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('native uint8 RGB required')
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=rgb.shape[:2]:
        raise ValueError('native uint8 alpha required')
    if alpha.size>8_000_000:raise ValueError('source exceeds bounded8MP copy budget')
    if digest(rgb)!=source_sha256 or digest(alpha)!=alpha_sha256:
        raise ValueError('source/alpha binding mismatch')
    if reviewed is not True or material not in ('opaque','opaque_fibres'):
        raise ValueError('explicit reviewed opaque region required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('bounded region review note required')
    if not isinstance(roi,(tuple,list)) or len(roi)!=4 or any(type(v)is not int for v in roi):
        raise ValueError('native integer ROI required')
    x0,y0,x1,y1=roi;h,w=alpha.shape
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('ROI outside source')
    if type(feather)is not int or not 1<=feather<=32 or 2*feather>=min(x1-x0,y1-y0):
        raise ValueError('bounded feather must fit native ROI')
    r=rgb[y0:y1,x0:x1];a=alpha[y0:y1,x0:x1]
    validate(r,a)
    tri=make_trimap(a,2) if trimap is None else trimap
    check_trimap(tri,a.shape)
    estimated_rgb,estimated_alpha,solver=refine_arrays(r,a,tri,method,edge_colors)
    yy,xx=np.indices(a.shape)
    distance=np.minimum.reduce([xx,yy,a.shape[1]-1-xx,a.shape[0]-1-yy])
    weight=np.minimum(1.,distance/feather)*(tri==128)
    blended_alpha=np.rint(a*(1-weight)+estimated_alpha*weight).astype(np.uint8)
    blended_rgb=np.rint(r*(1-weight[...,None])+estimated_rgb*weight[...,None]).astype(np.uint8)
    out_rgb=rgb.copy();out_alpha=alpha.copy()
    out_rgb[y0:y1,x0:x1]=blended_rgb;out_alpha[y0:y1,x0:x1]=blended_alpha
    # Invariants use only ROI-sized comparison arrays, not full-image float work.
    if not np.array_equal(blended_rgb[tri!=128],r[tri!=128]) or not np.array_equal(blended_alpha[tri!=128],a[tri!=128]):
        raise RuntimeError('definite trimap seeds changed')
    return out_rgb,out_alpha,dict(schema='native-region-matting.v1',source_sha256=source_sha256,
        prior_alpha_sha256=alpha_sha256,output_rgb_sha256=digest(out_rgb),output_alpha_sha256=digest(out_alpha),
        roi_xyxy=list(roi),roi_pixels=int(a.size),source_pixels=int(alpha.size),
        trimap_sha256=digest(tri),trimap_origin='supplied_reviewer_hypothesis' if trimap is not None else 'derived_2px_not_ground_truth',
        review_note=review_note,material=material,method=method,feather=feather,
        changed_rgb_pixels=int(np.any(blended_rgb!=r,2).sum()),changed_alpha_pixels=int((blended_alpha!=a).sum()),
        estimated_boundary_rgb=bool(edge_colors),source_rgb_exact=not bool(np.any(blended_rgb!=r)),
        outside_roi_rgb_alpha_exact=True,rectangle_border_rgb_alpha_exact=True,definite_trimap_seeds_exact=True,
        resampled=False,solver=solver,accepted=False,default_pipeline_changed=False,
        limitations=['Reviewer ROI/trimap are hypotheses, not semantic or optical truth.',
                    'Feathering retains the prior at the rectangle border and can retain its errors.',
                    'Boundary colour recovery is an estimate; product RGB is not globally source-exact when enabled.',
                    'No full-source isolation, hidden-detail recovery, relighting, camera or physical-contact claim.'])
