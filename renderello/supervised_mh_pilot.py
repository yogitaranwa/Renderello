"""Small supervised Magic Hour editor pilot. Never retries a POST.

A reservation is durable before sending. An interrupted/ambiguous request blocks
subsequent spending until a human reviews provider state. This module does not
verify owner authority: caller must have independently established that scope.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .credit_ledger import CreditLedger
from .matched_trial import _canonical, _identifier, _text, read_json

BASE = 'https://api.magichour.ai'
QUOTES_1K = {'gpt-image-2.5-flare':200,'nano-banana-pro':150,'nano-banana-2.1':50}
PRICE_SOURCE = 'https://static.magichour.ai/_next/static/chunks/056hftg9bmzl8.js'

def build_request(case_id, model, prompt, asset_path):
    _identifier(case_id);_text(prompt,'prompt',15000)
    if model not in QUOTES_1K:
        raise ValueError('model outside verified pilot prices')
    if not isinstance(asset_path,str) or not asset_path.startswith('api-assets/') or '..' in asset_path:
        raise ValueError('verified uploaded source asset required')
    return dict(name='Renderello matched pilot '+case_id+' '+model,
                image_count=1,model=model,aspect_ratio='1:1',resolution='1k',
                style=dict(prompt=prompt),assets=dict(image_file_paths=[asset_path]))

def submit_once(session, key, request, ledger, job, evidence_root, expected_account):
    _identifier(job)
    # No permissive defaults: compare exact current request with supported shape.
    model=request.get('model')
    if set(request) != {'name','image_count','model','aspect_ratio','resolution','style','assets'}:
        raise ValueError('unexpected request fields')
    if model not in QUOTES_1K or request['image_count'] != 1 or type(request['image_count']) is not int or request['resolution']!='1k' or request['aspect_ratio']!='1:1':
        raise ValueError('unquoted configuration refused')
    _text(request['name'],'name',200)
    if set(request['style'])!={'prompt'} or set(request['assets'])!={'image_file_paths'} or len(request['assets']['image_file_paths'])!=1:
        raise ValueError('unsupported request shape')
    build_request(job,model,request['style']['prompt'],request['assets']['image_file_paths'][0])
    root=Path(evidence_root);root.mkdir(parents=True,exist_ok=True)
    folder=root/job
    if folder.exists():
        raise FileExistsError('existing job: inspect evidence, never repeat submission')
    headers={'Authorization':'Bearer '+key}
    account=session.get(BASE+'/v1/account',headers=headers,timeout=30)
    account.raise_for_status();before=account.json()
    if before.get('id')!=expected_account:
        raise ValueError('current account differs from authorized project account')
    quote=QUOTES_1K[model]
    if type(before.get('credits')) is not int or before['credits']<quote:
        raise ValueError('insufficient or unverified available credits')
    folder.mkdir()
    (folder/'account-before.json').write_bytes(_canonical(before))
    request_bytes=_canonical(request)
    (folder/'request.json').write_bytes(request_bytes)
    ledger.reserve(job,quote,cost_source=PRICE_SOURCE,account_id=expected_account,
                   request_sha256=hashlib.sha256(request_bytes).hexdigest())
    (folder/'SUBMISSION_MAY_HAVE_OCCURRED').write_text('Reservation recorded before exactly one POST. Never retry blindly.\n')
    # Intentionally no exception recovery that could retry or erase a reservation.
    response=session.post(BASE+'/v1/ai-image-editor',headers=headers,json=request,timeout=90)
    (folder/'response-status.txt').write_text(str(response.status_code))
    (folder/'response.json').write_text(response.text)
    response.raise_for_status()
    body=response.json()
    if not isinstance(body.get('id'),str) or not body['id']:
        raise ValueError('missing project ID; reservation remains unresolved')
    ledger.submitted(job,body['id'])
    charged=body.get('credits_charged')
    if type(charged) is not int or charged<0:
        raise ValueError('missing actual charge; reservation remains unresolved')
    snapshot=ledger.settle(job,charged,receipt_reference=str(folder/'response.json'))
    (folder/'ledger-after.json').write_bytes(_canonical(snapshot))
    if charged!=quote or snapshot['over_cap']:
        raise ValueError('actual charge differs from verified quote; stop and review')
    return body
