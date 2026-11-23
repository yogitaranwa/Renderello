"""Explicit prepared-source bridge into CPU composition, not automatic quality approval.

Immutable input-byte snapshots prevent a changed prepared file from being used after
its hash check. Legacy entry points, publication and source-rights decisions stay
unchanged. This command never downloads a model or selects a paid backend.
"""
from pathlib import Path
from io import BytesIO
import hashlib,json,os,stat,tempfile
from PIL import Image
from .image_intake import IntakeError,IntakePolicy,_dimensions


def _read(path,limit):
    path=Path(path)
    if path.is_symlink():raise IntakeError('Prepared input symlink refused')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NONBLOCK',0)|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise IntakeError('Prepared input must be regular')
        if os.fstat(stream.fileno()).st_size>limit:raise IntakeError('Prepared input exceeds byte budget')
        data=stream.read(limit+1)
    if len(data)>limit:raise IntakeError('Prepared input exceeds byte budget')
    return data


def _pairs(rows):
    result={}
    for k,v in rows:
        if k in result:raise IntakeError('Duplicate prepared manifest key')
        result[k]=v
    return result


def load_prepared(folder):
    """Return verified bytes, not mutable paths. No writes on validation failure."""
    folder=Path(folder)
    if folder.is_symlink() or not folder.is_dir():raise IntakeError('Prepared directory required, no symlink')
    if (folder/'PREPARATION_INCOMPLETE').exists():raise IntakeError('Incomplete preparation refused')
    raw=_read(folder/'prepared-source.json',128_000)
    try:
        record=json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda x:(_ for _ in ()).throw(IntakeError('Nonfinite manifest value')))
        size=record['intake']['output_size']
        if not isinstance(size,list) or len(size)!=2 or any(type(x) is not int or x<1 for x in size):raise IntakeError('Invalid prepared dimensions')
    except (KeyError,TypeError,json.JSONDecodeError) as e:raise IntakeError('Invalid prepared manifest') from e
    files={}
    for name,mode,key in [('normalized-source.png','RGB','source_png_sha256'),('reviewed-alpha.png','L','alpha_png_sha256')]:
        data=_read(folder/name,40_000_000)
        if hashlib.sha256(data).hexdigest()!=record.get(key):raise IntakeError('Prepared bytes changed: '+name)
        if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)<29 or data[12:16]!=b'IHDR' or data[24]!=8:raise IntakeError('Expected 8-bit PNG')
        with Image.open(BytesIO(data)) as im:
            _dimensions(im,IntakePolicy())
            if im.format!='PNG' or im.mode!=mode or list(im.size)!=size or getattr(im,'n_frames',1)!=1:raise IntakeError('Prepared image mode or coordinates disagree')
            if im.getexif().get(274,1)!=1 or im.info.get('icc_profile') or 'transparency' in im.info:raise IntakeError('Hidden prepared interpretation metadata')
            im.load()
            if mode=='L' and im.getextrema()[1]==0:raise IntakeError('Prepared alpha empty')
        files[name]=data
    return record,files


def compose_prepared(folder,out_root,template='clean_studio',seed=0):
    from .templates import get_template
    from .config import RunConfig
    from .schemas import EditRequest,Route
    from .pipeline import run_pipeline
    if type(seed) is not int:raise ValueError('Integer seed required')
    selected=get_template(template)
    record,files=load_prepared(folder)
    out_root=Path(out_root);out_root.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix='prepared-compose-',dir=out_root));snap=job/'input';snap.mkdir()
    try:
        for name,data in files.items():(snap/name).write_bytes(data)
        # Retain provenance without upgrading the source rights or review status.
        (snap/'prepared-source.json').write_text(json.dumps(record,indent=2,allow_nan=False))
        request=EditRequest(snap/'normalized-source.png',selected,mask_path=snap/'reviewed-alpha.png')
        result=run_pipeline(request,[Route.CUTOUT_COMPOSE],RunConfig(variants=1,seed=seed,out_root=job/'runs'),relight_enabled=False)
        checks=_audit_outputs(result,files,selected)
        audit=dict(prepared_record=record,source_rights='NOT_UPGRADED',semantic_mask_accuracy='NOT_VERIFIED',publication_status='NOT_EVALUATED',legacy_routes_changed=False,relighting=False,paid_backend=False,run_manifest=str(result.manifest_path),composition_may_resize_source=True,pixel_fidelity='SEE_SCOPED_NATIVE_PIXEL_CHECKS',native_pixel_checks=checks)
        (job/'prepared-compose-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
        if any(check['status']=='MISMATCH' for check in checks):
            raise IntakeError('Saved composition changed fully opaque native RGB pixels')
        return result
    except BaseException:
        (job/'FAILED').touch();raise


def _audit_outputs(result,files,selected):
    """Check saved outputs, not a resampled QA baseline or only the input hash."""
    import numpy as np
    from .native_pixel_audit import audit_native_pixels
    from .pipeline import _placed_region
    from .schemas import CutoutResult
    manifest=json.loads(result.manifest_path.read_text())
    c=manifest['cutout']
    co=CutoutResult(Path(c['rgba_path']),Path(c['matte_path']),tuple(c['bbox']),c['backend'],c['coverage'])
    with Image.open(BytesIO(files['normalized-source.png'])) as im:source=np.array(im)
    with Image.open(BytesIO(files['reviewed-alpha.png'])) as im:alpha=np.array(im)
    checks=[]
    for variant in result.variants:
        raw=variant.output_path.read_bytes()
        with Image.open(BytesIO(raw)) as im:
            if im.mode!='RGB':raise IntakeError('Expected decoded RGB composition for native audit')
            output=np.array(im)
        region=_placed_region(co,selected.aspect,variant.seed,selected.key)
        check=audit_native_pixels(source,alpha,output,co.bbox,region)
        check.update(output=str(variant.output_path),output_sha256=hashlib.sha256(raw).hexdigest(),
                     placement_basis='EXPECTED_CPU_COMPOSITION_COORDINATES')
        checks.append(check)
    return checks


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description='Explicit CPU composition from verified prepared source/mask bytes; resizing may change pixels; no automatic quality approval')
    p.add_argument('--prepared',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--template',default='clean_studio');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args(argv);r=compose_prepared(a.prepared,a.out,a.template,a.seed);print(r.manifest_path)

if __name__=='__main__':main()
