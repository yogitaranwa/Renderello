"""Portable notebook helpers. Free preview by default; live work needs explicit scope.
No default key, account, credit allowance or approval is supplied by this module.
"""
from __future__ import annotations
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.parse import quote
from PIL import Image
from .credit_ledger import CreditLedger
from .supervised_mh_pilot import BASE, QUOTES_1K, build_request, submit_once


def validate_choices(template, instruction, alternatives):
    from .templates import get_template
    from .prompts import validate_instruction
    spec = get_template(template)
    validate_instruction(instruction)
    if type(alternatives) is not int or alternatives not in (2, 3, 4):
        raise ValueError('Choose 2, 3 or 4 alternatives')
    return spec


def preview(source, root, *, template='minimal_tabletop', instruction='',
            alternatives=2, material='unknown'):
    """CPU GrabCut composition, not generative AI or production-quality proof."""
    spec = validate_choices(template, instruction, alternatives)
    if material == 'transparent':
        raise ValueError('Free opaque preview is not a transparency workflow. Use reviewed optical inputs or a separately approved editor test.')
    from .source_contract import SourceContract
    from .cutout import cutout
    from .config import RunConfig
    from .schemas import EditRequest, Route
    from .pipeline import run_pipeline
    from .gallery import build_gallery
    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    import shutil
    stored_source = root / ('source-original' + source.suffix.lower())
    shutil.copyfile(source, stored_source)
    source = stored_source
    original = Image.open(source).convert('RGB')
    # Explicit backend prevents incidental model downloads on a Colab runtime.
    mask = cutout(original, root / 'mask', backend='grabcut', device='cpu')
    result = run_pipeline(EditRequest(source, spec, custom_instruction=instruction,
        mask_path=mask.matte_path), [Route.CUTOUT_COMPOSE],
        RunConfig(variants=alternatives, out_root=root / 'runs'),
        relight_enabled=False, source_contract=SourceContract(material=material))
    gallery = build_gallery([result.run_dir], root / 'gallery.html')
    return {'root': str(root), 'gallery': str(gallery),
        'outputs': [str(v.output_path) for v in result.variants if v.output_path.exists()],
        'manifest': str(result.manifest_path),
        'kind': 'FREE_CPU_PROCEDURAL_PREVIEW_NOT_AI_EDITOR_QUALITY'}


def submit_live(source, root, *, session, key, expected_account, cap,
                authority_reference, source_upload_approved=False,
                spending_approved=False, quote_confirmed=False,
                template='minimal_tabletop', instruction='', alternatives=2,
                model='nano-banana-2.1'):
    """One new batch only. Calls cannot retry an existing folder.

    These booleans record a caller declaration, not verified consent. A notebook
    must be operated by the account owner or under independently checked scope.
    Quotes are the 11 Nov 2026 1k-square snapshot, not a live pricing service.
    """
    spec = validate_choices(template, instruction, alternatives)
    if not all(x is True for x in (source_upload_approved, spending_approved, quote_confirmed)):
        raise ValueError('Source upload, spending and current quote confirmations are all required')
    if model not in QUOTES_1K or not expected_account or not key:
        raise ValueError('Supported model, intended account and runtime key required')
    if type(cap) is not int or cap < alternatives * QUOTES_1K[model]:
        raise ValueError('Explicit batch credit limit is below the quoted cost')
    if not authority_reference or not authority_reference.strip():
        raise ValueError('Record the scope of owner approval; this field grants no authority')
    from .schemas import EditRequest, Aspect
    from .prompts import build_prompt
    source, root = Path(source), Path(root)
    # Verify source and account before uploading anything.
    image = Image.open(source); image.verify()
    headers = {'Authorization': 'Bearer ' + key}
    response = session.get(BASE + '/v1/account', headers=headers, timeout=30)
    response.raise_for_status()
    account = response.json()
    if account.get('id') != expected_account or type(account.get('credits')) is not int:
        raise ValueError('Account identity or credit balance does not match')
    if account['credits'] < alternatives * QUOTES_1K[model]:
        raise ValueError('Account balance below batch quote')
    root.mkdir(parents=True, exist_ok=False)
    data = source.read_bytes()
    (root / ('source-original' + source.suffix.lower())).write_bytes(data)  # bytes preserved; actual type retained by PIL
    ledger = CreditLedger(root / 'credit-ledger', cap=cap, authority_reference=authority_reference)
    prompt = build_prompt(EditRequest(source, spec, custom_instruction=instruction, aspect=Aspect.SQUARE))
    plan = {'source_sha256': hashlib.sha256(data).hexdigest(), 'template': template,
        'prompt': prompt, 'alternatives': alternatives, 'model': model,
        'quote_per_output': QUOTES_1K[model], 'quote_total': alternatives * QUOTES_1K[model],
        'quote_date': '2026-11-11', 'resolution': '1k', 'aspect': '1:1',
        'approval_reference': authority_reference, 'submitted_at': time.time(),
        'caller_confirmations_are_not_authentication': True}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    # Normalize uploaded encoding explicitly while preserving saved original bytes.
    import io
    buf = io.BytesIO(); Image.open(source).convert('RGB').save(buf, 'PNG')
    up = session.post(BASE + '/v1/files/upload-urls', headers=headers,
        json={'items': [{'type': 'image', 'extension': 'png'}]}, timeout=30)
    up.raise_for_status(); item = up.json()['items'][0]
    put = session.put(item['upload_url'], data=buf.getvalue(),
        headers={'Content-Type': 'image/png'}, timeout=60)
    put.raise_for_status()
    for i in range(alternatives):
        job = 'alternative-' + str(i + 1)
        request = build_request(job, model, prompt, item['file_path'])
        submit_once(session, key, request, ledger, job, root / 'jobs', expected_account)
    return {'root': str(root), 'ledger': ledger.snapshot(), 'status': 'SUBMITTED_NOT_COMPLETE'}


def recover_live(root, *, session, key, minimum_interval=3600):
    """Read existing IDs once. Never create an image; default hourly checks."""
    root = Path(root)
    stamp = root / 'last-check.json'
    if stamp.exists() and time.time() - json.loads(stamp.read_text())['unix'] < minimum_interval:
        raise ValueError('Wait until the hourly check is due; no faster automatic polling')
    stamp.write_text(json.dumps({'unix': time.time()}))
    headers = {'Authorization': 'Bearer ' + key}
    results = []
    for folder in sorted((root / 'jobs').iterdir()):
        if not folder.is_dir(): continue
        if not (folder / 'response.json').exists():
            raise ValueError('Submission outcome unknown. Inspect account/provider; never resubmit blindly')
        project_id = json.loads((folder / 'response.json').read_text())['id']
        response = session.get(BASE + '/v1/image-projects/' + quote(project_id, safe=''),
            headers=headers, timeout=30)
        response.raise_for_status(); detail = response.json()
        (folder / 'latest-detail.json').write_text(json.dumps(detail, indent=2))
        output = folder / 'output.png'
        if detail.get('status') == 'complete' and not output.exists():
            downloads = detail.get('downloads') or []
            if not downloads: raise ValueError('Complete project has no download; preserve ID')
            downloaded = session.get(downloads[0]['url'], timeout=90)
            downloaded.raise_for_status()
            import io
            Image.open(io.BytesIO(downloaded.content)).verify()
            output.write_bytes(downloaded.content)
        results.append({'job': folder.name, 'project_id': project_id,
            'status': detail.get('status'), 'credits_charged': detail.get('credits_charged'),
            'output': str(output) if output.exists() else None})
    return results
