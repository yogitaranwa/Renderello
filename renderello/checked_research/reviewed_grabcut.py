"""Explicit manual native GrabCut research. No default backend or quality pass."""
from pathlib import Path
import hashlib,json,time
import cv2
import numpy as np
from PIL import Image

CLASSES={'probable_foreground':cv2.GC_PR_FGD,'definite_foreground':cv2.GC_FGD,'definite_background':cv2.GC_BGD}

def seed_map(data, source_bytes, size):
    if not isinstance(data,dict) or data.get('source_sha256')!=hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('exact source file SHA256 required')
    if data.get('source_size')!=list(size):raise ValueError('source dimensions differ')
    if data.get('origin')!='assistant visual review coarse manual polygons, not ground truth':
        raise ValueError('explicit manual research origin required')
    coord=data.get('annotation_coordinate_space')
    if not isinstance(coord,list) or len(coord)!=2 or any(type(x)is not int or x<2 for x in coord):raise ValueError('integer coordinate dimensions required')
    w,h=size;cw,ch=coord
    if w*ch!=h*cw:raise ValueError('annotation aspect differs')
    result=np.zeros((h,w),np.uint8)
    for key,cls in CLASSES.items():
        polys=data.get(key)
        if not isinstance(polys,list) or not 1<=len(polys)<=32:raise ValueError('one to32polygons per class required')
        for poly in polys:
            a=np.asarray(poly)
            if a.dtype.kind not in 'fiu' or a.ndim!=2 or a.shape[1]!=2 or not 3<=len(a)<=128 or not np.isfinite(a).all():raise ValueError('bounded finite polygon required')
            if np.any(a<0) or np.any(a[:,0]>=cw) or np.any(a[:,1]>=ch):raise ValueError('polygon outside reviewed coordinate frame')
            q=np.rint(a*np.array([w/cw,h/ch])).astype(np.int32)
            if cv2.contourArea(q)<1:raise ValueError('degenerate polygon')
            cv2.fillPoly(result,[q],cls)
    if any(not np.any(result==x) for x in (cv2.GC_FGD,cv2.GC_BGD,cv2.GC_PR_FGD)):raise ValueError('all foreground/background constraints must survive precedence')
    return result

def run(source,seeds,out):
    source,seeds,out=map(Path,(source,seeds,out))
    if out.exists() or out.is_symlink():raise FileExistsError('preserve prior research output')
    if not source.is_file() or source.is_symlink() or source.stat().st_size>32_000_000:raise ValueError('bounded regular source required')
    raw=source.read_bytes();data=json.loads(seeds.read_text())
    with Image.open(source) as im:
        if im.mode!='RGB' or getattr(im,'n_frames',1)!=1 or not 9<=im.width*im.height<=4_000_000:raise ValueError('native single-frame RGB3px..4MP required')
        if im.info.get('icc_profile') or im.getexif().get(274,1)!=1:raise ValueError('explicit color/orientation normalization required')
        rgb=np.array(im);size=im.size
    classes=seed_map(data,raw,size);initial=classes.copy()
    cv2.setNumThreads(1);cv2.setRNGSeed(1182026);start=time.monotonic()
    cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),classes,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
    alpha=np.where((classes==cv2.GC_FGD)|(classes==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    if np.any(alpha[initial==cv2.GC_FGD]!=255) or np.any(alpha[initial==cv2.GC_BGD]!=0):raise RuntimeError('manual hard seeds changed')
    out.mkdir(parents=True)
    Image.fromarray(alpha).save(out/'alpha.png');Image.fromarray(np.dstack([rgb,alpha])).save(out/'rgba.png')
    saved=np.array(Image.open(out/'rgba.png'))
    if not np.array_equal(saved[:,:,:3],rgb):raise RuntimeError('native source RGB changed')
    # No scene or lighting is synthesized by this diagnostic.
    for name,color in [('white',(255,255,255)),('magenta',(190,0,150))]:
        preview=np.where(alpha[:,:,None]>0,rgb,np.array(color,np.uint8));Image.fromarray(preview).save(out/(name+'-check.png'))
    report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),seed_sha256=hashlib.sha256(seeds.read_bytes()).hexdigest(),opencv=cv2.__version__,seconds=time.monotonic()-start,source_size=list(size),source_RGB_changed=0,source_resampled=False,paid_calls=0,manual_supervision=True,mask_is_binary_not_optical=True,semantic_completeness='NOT_GROUND_TRUTH',quality_pass=False,default_backend_changed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));return report
