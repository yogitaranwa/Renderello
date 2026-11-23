"""Explicit source-bound opaque recovery from an inspected semantic silhouette.
Not automatic acceptance, not a union of guesses, not transparent-object matting.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .intent_mask import sha,load_bundle
from .prompted_mask import validate_prompts,MODEL_SHA256
from .matting import guarded_file


def verify_reviewed(source,reviewed,expected_alpha_sha256):
    with Image.open(source) as im:
        if im.width*im.height>1_000_000:raise ValueError('opaque recovery capped at one million pixels; no resize')
        pixels=np.array(im.convert('RGB'));size=im.size
    manifest,alpha=load_bundle(reviewed,'prompted-mask-manifest.json',sha(source),size)
    if sha(reviewed/'estimated-alpha.png')!=expected_alpha_sha256:raise ValueError('reviewed silhouette hash mismatch')
    if manifest.get('model_sha256')!=MODEL_SHA256:raise ValueError('unsupported reviewed model')
    pp=reviewed/'reviewed-prompts.json'
    if sha(pp)!=manifest.get('prompt_sha256'):raise ValueError('prompt hash mismatch')
    objects=validate_prompts(json.loads(pp.read_text()),source,*size)
    if manifest.get('prompt_origin')!='reviewed_source_geometry' or manifest.get('objects')!=[o['name'] for o in objects]:raise ValueError('reviewed provenance mismatch')
    if any(1 not in o['labels'] for o in objects):raise ValueError('explicit positive points required on every product')
    if not np.isin(alpha,[0,255]).all() or not np.any(alpha==0):raise ValueError('reviewed silhouette must be binary and include background')
    with Image.open(reviewed/'source.png') as im:
        if not np.array_equal(np.array(im.convert('RGB')),pixels):raise ValueError('reviewed source RGB mismatch')
    with Image.open(reviewed/'cutout.png') as im:
        rgba=np.array(im)
        if im.mode!='RGBA' or not np.array_equal(rgba[:,:,:3],pixels) or not np.array_equal(rgba[:,:,3],alpha):raise ValueError('reviewed cutout mismatch')
    return manifest


def recover_file(source:Path,reviewed:Path,out:Path,*,reviewed_alpha_sha256,material,framing,confirm_reviewed_silhouette=False,method='cf',radius=3):
    if out.exists():raise ValueError('output exists; preserve attempts')
    if material!='opaque':raise ValueError('opaque-only recovery; transparent or unknown material requires a different workflow')
    if framing!='complete':raise ValueError('complete source required; this cannot recover cropped parts')
    if not confirm_reviewed_silhouette:raise ValueError('explicit inspection of this exact silhouette required')
    if method not in ('cf','lkm'):raise ValueError('opaque recovery requires cf or lkm')
    verify_reviewed(source,reviewed,reviewed_alpha_sha256)
    # Different trust decision from min-alpha gate: use reviewed silhouette as
    # initial semantic support, never lock the known-bad automatic alpha as BG.
    result=guarded_file(source,reviewed/'estimated-alpha.png',out,method=method,radius=radius,edge_colors=False)
    result.update(route='explicit_reviewed_opaque_recovery',initial_support='inspected_HQ_SAM_silhouette_NOT_automatic_alpha',reviewed_manifest_sha256=sha(reviewed/'prompted-mask-manifest.json'),reviewed_alpha_sha256=reviewed_alpha_sha256,material_declaration=material,framing_declaration=framing,silhouette_confirmation='explicit_caller_confirmation_not_proof_of_human_review',quality='NOT_EVALUATED',fine_print='NOT_INDEPENDENTLY_VERIFIED',warnings=result['warnings']+['This route can restore source pixels missing from an automatic saliency mask only if the supplied reviewed silhouette includes them.','The trimap is still heuristic; it can lock incorrect foreground and background. Inspect final edges on all backgrounds.','Opaque material and complete framing are caller declarations, not inferred facts. No glass transmission or uncropping.'])
    (out/'opaque-recovery-manifest.json').write_text(json.dumps(result,indent=2));return result
