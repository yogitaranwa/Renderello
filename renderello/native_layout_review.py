"""Unregistered native-layout review packs, without stretching product geometry.

Unlike blind_review's shared-coordinate packs, these packs allow different native
sizes. They preserve original encoded bytes and prohibit pixelwise/ROI alignment
claims. Tokenization does not prove independence or hide recognizable methods.
Verification detects accidental changes against COMPLETE.json, not authenticity
against a party able to replace both the files and the manifest.
"""
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil
from .blind_review import Candidate
from .image_intake import IntakePolicy, inspect_image
from .review import DIMENSIONS


def digest(path):
    h=hashlib.sha256();total=0
    with Path(path).open('rb') as f:
        while True:
            chunk=f.read(1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>40_000_000:raise ValueError('Review input exceeds encoded byte budget')
            h.update(chunk)
    return h.hexdigest()


def build_native_pack(source, candidates, out, *, seed):
    source=Path(source);out=Path(out)
    if out.exists():raise FileExistsError('Never replace an earlier review')
    if type(seed) is not int:raise ValueError('Integer seed required')
    if not 2<=len(candidates)<=12:raise ValueError('Need2..12 candidates')
    if any(not isinstance(c,Candidate) for c in candidates):raise ValueError('Typed candidate required')
    if len({c.method for c in candidates})!=len(candidates):raise ValueError('Duplicate method')
    policy=IntakePolicy(allow_colour_conversion=True)
    def inspect(path):
        path=Path(path)
        if path.is_symlink() or not path.is_file():raise ValueError('Regular nonsymlink input required')
        before=digest(path);image=inspect_image(path,policy)
        try:return {'path':path,'sha256':before,'size':list(image.record['input_size']),'display_size':list(image.image.size),'mode':image.image.mode,'format':image.record['input_format']}
        finally:image.close()
    src=inspect(source);records=[];seen={src['sha256']}
    order=list(candidates);random.Random(seed).shuffle(order)
    for c in order:
        if c.path is None:records.append((c,None));continue
        info=inspect(c.path)
        if info['sha256'] in seen:raise ValueError('Duplicate source/output content')
        seen.add(info['sha256']);records.append((c,info))
    # Validate every input before creating the output, then recheck during copy.
    for info in [src,*[i for _,i in records if i]]:
        if digest(info['path'])!=info['sha256']:raise ValueError('Input changed during preparation')
    out.mkdir(parents=True);review=out/'REVIEW';review.mkdir();private=out/'PRIVATE';private.mkdir()
    files={};key=[];rows=[]
    suffix={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp','TIFF':'.tiff','BMP':'.bmp'}
    def copy(info,token):
        name=token+suffix[info['format']];target=review/name
        shutil.copyfile(info['path'],target)
        if digest(target)!=info['sha256'] or digest(info['path'])!=info['sha256']:
            raise ValueError('Input changed while copying; pack incomplete')
        files[name]=info['sha256'];return name
    source_name=copy(src,'source')
    for index,(candidate,info) in enumerate(records,1):
        token=f'C{index:02d}';name=copy(info,token) if info else None
        key.append({'token':token,'method':candidate.method,'source_path':str(candidate.path) if candidate.path else None,'failure':candidate.failure})
        rows.append({'token':token,'file':name,'native_size':info['size'] if info else None,'status':'NATIVE_ARTIFACT_PRESENT' if info else 'MISSING_OUTPUT','failure':candidate.failure if not info else ''})
    manifest={'schema':'native-layout-review.v1','source_file':source_name,'source_size':src['size'],'candidates':rows,
      'registration':'NOT_ESTABLISHED','pixelwise_comparison_permitted':False,
      'native_encoded_bytes_preserved':True,'independent_blinding':'NOT_ESTABLISHED',
      'recognizable_methods_may_be_known':True,'ratings':'BLANK','acceptance':'NOT_EVALUATED'}
    (review/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (review/'scores-blank.csv').open('w',newline='') as f:
        fields=['token',*DIMENSIONS,'no_critical','critical_errors','provenance','notes'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({'token':r['token']} for r in rows)
    (private/'key.json').write_text(json.dumps({'seed':seed,'source_path':str(source),'candidates':key},indent=2))
    for p in review.iterdir():
        if p.is_file():files[p.name]=digest(p)
    (out/'COMPLETE.json').write_text(json.dumps({'review_sha256':files,'outputs':sum(i is not None for _,i in records),'planned':len(records)},indent=2))
    return out


def verify_native_pack(out):
    out=Path(out);record=json.loads((out/'COMPLETE.json').read_text());review=out/'REVIEW'
    hashes=record['review_sha256']
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Empty manifest')
    if review.is_symlink() or {p.name for p in review.iterdir()}!=set(hashes):raise ValueError('Unexpected review contents')
    for name,sha in hashes.items():
        path=review/name
        if Path(name).name!=name or path.is_symlink() or not path.is_file():raise ValueError('Unsafe review path')
        if digest(path)!=sha:raise ValueError('Review content changed')
    return record
