"""Explicit reviewed intent gate for an existing automatic alpha. No inference or acceptance."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from .prompted_mask import validate_prompts,MODEL_SHA256 as HQ_SHA256

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_bundle(folder,manifest_name,source_hash,size):
    m=json.loads((folder/manifest_name).read_text())
    if m.get('source_sha256')!=source_hash:raise ValueError('bundle source SHA256 mismatch')
    paths=('estimated-alpha.png','source.png','cutout.png')
    for name in paths:
        if m.get('output_sha256',{}).get(name)!=sha(folder/name):raise ValueError('bundle output SHA256 mismatch: '+name)
    with Image.open(folder/'estimated-alpha.png') as im:
        if im.mode!='L' or im.size!=size:raise ValueError('alpha must be source-sized L image')
        a=np.array(im)
    if not np.any(a):raise ValueError('empty input alpha')
    return m,a

def gate_file(source:Path,automatic:Path,reviewed:Path,out:Path):
    if out.exists():raise ValueError('output exists; preserve attempts')
    with Image.open(source) as im:
        if im.width*im.height>4_000_000:raise ValueError('source exceeds4million pixels')
        rgb=im.convert('RGB');pixels=np.array(rgb);size=im.size
    source_hash=sha(source)
    candidates=[n for n in ('bilateral-mask-manifest.json','pyramid-mask-manifest.json','semantic-mask-manifest.json') if (automatic/n).is_file()]
    if len(candidates)!=1:raise ValueError('exactly one supported automatic manifest required')
    auto,a=load_bundle(automatic,candidates[0],source_hash,size)
    prompt,b=load_bundle(reviewed,'prompted-mask-manifest.json',source_hash,size)
    if prompt.get('model_sha256')!=HQ_SHA256:raise ValueError('unsupported reviewed HQ-SAM model')
    pp=reviewed/'reviewed-prompts.json'
    if sha(pp)!=prompt.get('prompt_sha256'):raise ValueError('reviewed prompt SHA256 mismatch')
    data=json.loads(pp.read_text());objects=validate_prompts(data,source,*size)
    if prompt.get('prompt_origin')!='reviewed_source_geometry' or prompt.get('objects')!=[o['name'] for o in objects]:raise ValueError('reviewed intent provenance mismatch')
    if not np.isin(b,[0,255]).all():raise ValueError('reviewed mask must be binary')
    for folder,expected_alpha in ((automatic,a),(reviewed,b)):
        with Image.open(folder/'source.png') as im:
            if not np.array_equal(np.array(im.convert('RGB')),pixels):raise ValueError('bundle decoded RGB mismatch')
        with Image.open(folder/'cutout.png') as im:
            if im.mode!='RGBA' or not np.array_equal(np.array(im)[:,:,:3],pixels):raise ValueError('bundle cutout RGB mismatch')
            if not np.array_equal(np.array(im)[:,:,3],expected_alpha):raise ValueError('bundle cutout alpha mismatch')
    alpha=np.minimum(a,b)
    if not np.any(alpha):raise ValueError('empty intersection; incompatible masks')
    out.mkdir(parents=True)
    for name,v in [('source.png',pixels),('estimated-alpha.png',alpha),('automatic-alpha.png',a),('reviewed-alpha.png',b),('cutout.png',np.dstack([pixels,alpha]))]:Image.fromarray(v).save(out/name)
    for name,col in [('black',(20,20,20)),('white',(250,250,250)),('magenta',(190,0,150))]:
        bg=Image.new('RGB',size,col);bg.paste(rgb,mask=Image.fromarray(alpha));bg.save(out/(name+'-review.png'))
    (out/'reviewed-prompts.json').write_bytes(pp.read_bytes())
    report=dict(source_sha256=source_hash,source_rgb_unchanged=True,method='minimum(automatic alpha, reviewed binary HQ-SAM intent)',automatic_manifest_sha256=sha(automatic/candidates[0]),reviewed_manifest_sha256=sha(reviewed/'prompted-mask-manifest.json'),prompt_sha256=sha(pp),objects=[o['name'] for o in objects],input_automatic_model=auto.get('model'),removed_nonzero_pixels=int(np.count_nonzero((a>0)&(alpha==0))),retained_alpha_values_exact=True,quality='NOT_EVALUATED',semantic_completeness='NOT_VERIFIED',source_framing='NOT_VERIFIED',warnings=['Manual reviewed intent, not an automatic success. Boxes/points need source review.','Intersection only removes; it cannot recover any automatic-mask missing pixels.','HQ-SAM errors and hard jagged boundaries transfer into the intersection. No optical glass matting.','Source RGB stays exact, including source-color fringe. No quality acceptance or default switch.'])
    report['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'intent-mask-manifest.json').write_text(json.dumps(report,indent=2));return report
