"""Local explicit reviewed-mask pipeline. No network or automatic checkpoint download."""
from pathlib import Path
import io,json,zipfile,uuid
from .prompted_mask import validate_prompts,guarded_file
from .intent_mask import gate_file


def unpack_automatic(data,folder):
    if not data or len(data)>20_000_000:raise ValueError('automatic bundle ZIP cap20MB')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos=z.infolist()
        if len(infos)>20 or sum(i.file_size for i in infos)>40_000_000:raise ValueError('bundle exceeds20files/40MB unpacked')
        required={'source.png','cutout.png','estimated-alpha.png'}
        manifests={'bilateral-mask-manifest.json','pyramid-mask-manifest.json','semantic-mask-manifest.json'}
        names=[i.filename for i in infos]
        if any('/' in n or '\\' in n or n.startswith('.') for n in names) or len(set(names))!=len(names):raise ValueError('flat unique filenames required, no directories')
        found=manifests.intersection(names)
        if len(found)!=1 or not required.issubset(names):raise ValueError('missing automatic manifest/alpha/source/cutout')
        folder.mkdir()
        for name in required|found:
            info=z.getinfo(name)
            if info.flag_bits&1:raise ValueError('encrypted bundle unsupported')
            (folder/name).write_bytes(z.read(name))


def run_pipeline(out_root,job,prompts,bundle,checkpoint):
    import re
    if checkpoint is None or not Path(checkpoint).is_file():raise ValueError('server checkpoint not configured; see UI startup instructions')
    if not re.fullmatch(r'prompt-[a-f0-9]{12}',job):raise ValueError('invalid source job')
    prepared=out_root/job;source=prepared/'original-source.bin'
    if not source.is_file():raise ValueError('prepared source not found')
    from PIL import Image
    with Image.open(source) as im:validate_prompts(prompts,source,*im.size)
    result=out_root/('intent-'+uuid.uuid4().hex[:12]);result.mkdir()
    try:
        unpack_automatic(bundle,result/'automatic')
        pp=result/'reviewed-prompts.json';pp.write_text(json.dumps(prompts,indent=2))
        guarded_file(source,result/'reviewed',pp,Path(checkpoint))
        report=gate_file(source,result/'automatic',result/'reviewed',result/'combined')
        # Compact downloadable bundle preserves the inputs, prompts and reviews.
        archive=result/'reviewed-mask-result.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in result.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,str(p.relative_to(result)))
        return dict(job=result.name,report=report,preview_urls={color:'/file/'+result.name+'/combined/'+color+'-review.png' for color in ('white','black','magenta')},download_url='/file/'+result.name+'/reviewed-mask-result.zip')
    except Exception as e:
        (result/'FAILED.txt').write_text(type(e).__name__+': '+str(e))
        raise


def _snapshot(folder):
    from .intent_mask import sha
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in {'preview-state.json','FAILED.txt'}}


def preview_pipeline(out_root,job,prompts,bundle,checkpoint):
    """Infer only. No intersection until a separate exact-preview confirmation."""
    import re,hashlib
    from .intent_mask import load_bundle,sha
    from PIL import Image
    if checkpoint is None or not Path(checkpoint).is_file():raise ValueError('server checkpoint not configured')
    if not re.fullmatch(r'prompt-[a-f0-9]{12}',job):raise ValueError('invalid source job')
    source=out_root/job/'original-source.bin'
    if not source.is_file():raise ValueError('prepared source not found')
    with Image.open(source) as im:
        objects=validate_prompts(prompts,source,*im.size);size=im.size
    if any(1 not in o['labels'] for o in objects):raise ValueError('at least one explicit positive product point per object is required for UI preview; points do not guarantee quality')
    result=out_root/('intent-'+uuid.uuid4().hex[:12]);result.mkdir()
    try:
        unpack_automatic(bundle,result/'automatic')
        manifest=next(p.name for p in (result/'automatic').glob('*manifest.json'))
        load_bundle(result/'automatic',manifest,sha(source),size)
        (result/'original-source.bin').write_bytes(source.read_bytes())
        pp=result/'reviewed-prompts.json';pp.write_text(json.dumps(prompts,indent=2))
        guarded_file(result/'original-source.bin',result/'reviewed',pp,Path(checkpoint))
        files=_snapshot(result);fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        (result/'preview-state.json').write_text(json.dumps(dict(files=files,fingerprint=fingerprint,state='AWAITING_EXPLICIT_SILHOUETTE_REVIEW'),indent=2))
        return dict(job=result.name,fingerprint=fingerprint,stage='REVIEW_SILHOUETTE_NOT_COMBINED',preview_urls={c:'/file/'+result.name+'/reviewed/'+c+'-review.png' for c in ('white','black','magenta')})
    except Exception as e:
        (result/'FAILED.txt').write_text(type(e).__name__+': '+str(e));raise


def finish_pipeline(out_root,job,fingerprint,acceptance):
    import re
    if acceptance!='I inspected this exact inferred silhouette':raise ValueError('explicit silhouette acceptance required')
    if not re.fullmatch(r'intent-[a-f0-9]{12}',job):raise ValueError('invalid preview job')
    result=out_root/job;state=json.loads((result/'preview-state.json').read_text())
    if state.get('state')!='AWAITING_EXPLICIT_SILHOUETTE_REVIEW':raise ValueError('preview is not awaiting review; preserve existing result')
    if fingerprint!=state.get('fingerprint') or _snapshot(result)!=state.get('files'):raise ValueError('preview inputs or images changed; infer and review again')
    report=gate_file(result/'original-source.bin',result/'automatic',result/'reviewed',result/'combined')
    state['state']='EXPLICIT_SILHOUETTE_REVIEW_CONFIRMED_NOT_QUALITY_ACCEPTED';(result/'preview-state.json').write_text(json.dumps(state,indent=2))
    archive=result/'reviewed-mask-result.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in result.rglob('*'):
            if p.is_file() and p!=archive:z.write(p,str(p.relative_to(result)))
    return dict(job=result.name,report=report,preview_urls={c:'/file/'+result.name+'/combined/'+c+'-review.png' for c in ('white','black','magenta')},download_url='/file/'+result.name+'/reviewed-mask-result.zip')
