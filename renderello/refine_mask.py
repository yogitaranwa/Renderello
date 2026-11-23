"""Explicit CPU seed-map correction. Human/engineer input, never automatic success.
Seed values: 0 definite BG,64 probable BG,128 unchanged,192 probable FG,255 definite FG.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def _gray(path:Path,size:tuple[int,int])->np.ndarray:
    with Image.open(path) as im:
        if im.mode!='L' or im.size!=size:
            raise ValueError('alpha and seed map must be source-sized 8-bit grayscale; no resize')
        return np.array(im)


def refine(source:Path,alpha:Path,seeds:Path,out:Path,iters:int=5)->dict:
    if out.exists():raise ValueError('output directory exists; use a fresh directory')
    if not 1<=iters<=8:raise ValueError('iterations must be1-8')
    with Image.open(source) as image:im=np.array(image.convert('RGB'))
    if im.shape[0]*im.shape[1]>4_000_000:raise ValueError('seed refinement capped at4million source pixels')
    size=(im.shape[1],im.shape[0]);a=_gray(alpha,size);s=_gray(seeds,size)
    if not set(np.unique(s)).issubset({0,64,128,192,255}):raise ValueError('seed values must be0,64,128,192,255')
    if not np.any(s==0) or not np.any(s==255):raise ValueError('explicit definite foreground AND background seeds required')
    m=np.where(a>127,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8')
    m[a>245]=cv2.GC_FGD
    for value,label in [(0,cv2.GC_BGD),(64,cv2.GC_PR_BGD),(192,cv2.GC_PR_FGD),(255,cv2.GC_FGD)]:m[s==value]=label
    t=time.monotonic()
    cv2.grabCut(cv2.cvtColor(im,cv2.COLOR_RGB2BGR),m,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),iters,cv2.GC_INIT_WITH_MASK)
    result=np.where((m==1)|(m==3),255,0).astype('uint8')
    if not np.any(result>127) or not np.any(result<128):raise ValueError('refinement produced empty/full mask')
    # Only one pixel blur, then restore exact definite labels. No color correction.
    result=cv2.GaussianBlur(result,(3,3),0);result[s==0]=0;result[s==255]=255
    out.mkdir(parents=True)
    Image.fromarray(a).save(out/'initial-alpha.png');Image.fromarray(s).save(out/'seed-map.png')
    Image.fromarray(result).save(out/'refined-alpha.png');rgba=Image.fromarray(im);rgba.putalpha(Image.fromarray(result));rgba.save(out/'refined-cutout.png')
    bg=Image.new('RGB',size,(218,236,223));bg.paste(Image.fromarray(im),(0,0),Image.fromarray(result));bg.save(out/'review-preview.png')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'status':'REVIEW_REQUIRED','backend':'seeded_grabcut_assisted','source':str(source.resolve()),
              'source_sha256':digest(source),'initial_alpha_sha256':digest(alpha),'seed_map_sha256':digest(seeds),
              'alpha_sha256':digest(out/'refined-alpha.png'),'alpha':'refined-alpha.png',
              'iterations':iters,'seconds':time.monotonic()-t,
              'warning':'Engineer/human-seeded correction, not automatic model success or recovered GPU alpha. Inspect clipping, missing products, halos and frame cuts.'}
    (out/'mask-manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','alpha','seeds','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iters',type=int,default=5);a=p.parse_args()
    print(json.dumps(refine(a.source,a.alpha,a.seeds,a.out,a.iters)));return 0

if __name__=='__main__':raise SystemExit(main())
