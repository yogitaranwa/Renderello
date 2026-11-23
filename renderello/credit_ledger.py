"""Append-only local credit reservations. Not a grant of spending authority.

Reserve a source-grounded quote BEFORE submission. A possibly submitted request
keeps its full reservation until actual charge or verified non-submission is known.
No retry, top-up, API call or permission inference is performed here.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from .matched_trial import _canonical, _identifier, _number, read_json


class CreditLedger:
    def __init__(self, root, *, cap=None, authority_reference=None):
        self.root = Path(root)
        self.meta = self.root / 'budget.json'
        if not self.root.exists():
            if type(cap) is not int or cap < 0:
                raise ValueError('explicit nonnegative integer approved cap required')
            if not isinstance(authority_reference, str) or not authority_reference.strip():
                raise ValueError('original approval reference required; this does not authenticate it')
            self.root.mkdir(parents=True)
            self.meta.write_bytes(_canonical(dict(cap=cap, authority_reference=authority_reference)))
        if self.root.is_symlink():
            raise ValueError('ledger symlink refused')
        self.config = read_json(self.meta)
        if type(self.config.get('cap')) is not int or self.config['cap'] < 0:
            raise ValueError('invalid ledger cap')
        if cap is not None and cap != self.config['cap']:
            raise ValueError('cannot silently change approved cap')

    def _read(self):
        events = []
        previous = '0' * 64
        paths = sorted(self.root.glob('event-*.json'))
        for i, p in enumerate(paths, 1):
            if p.name != f'event-{i:06d}.json':
                raise ValueError('missing credit-ledger event')
            e = read_json(p)
            digest = e.pop('digest')
            if e.get('previous') != previous or hashlib.sha256(_canonical(e)).hexdigest() != digest:
                raise ValueError('credit-ledger hash chain mismatch')
            previous = digest
            events.append(e)
        return events, previous

    def _state(self):
        events, previous = self._read()
        jobs = {}
        for e in events:
            ident = e['job']
            if e['kind'] == 'reserve':
                if ident in jobs:
                    raise ValueError('duplicate reservation')
                jobs[ident] = dict(e, status='reserved', charged=None)
            elif e['kind'] == 'submitted':
                if ident not in jobs or jobs[ident]['status'] != 'reserved':
                    raise ValueError('submission state conflict')
                jobs[ident].update(status='submitted', project_id=e['project_id'])
            elif e['kind'] == 'settle':
                if ident not in jobs or jobs[ident]['status'] not in {'reserved', 'submitted'}:
                    raise ValueError('settlement state conflict')
                jobs[ident].update(status='settled', charged=e['charged'])
            else:
                raise ValueError('unknown ledger event')
        return events, previous, jobs

    def snapshot(self):
        _, _, jobs = self._state()
        committed = sum(j['charged'] if j['status'] == 'settled' else j['quote'] for j in jobs.values())
        return dict(cap=self.config['cap'], committed_or_reserved=committed,
                    remaining=self.config['cap'] - committed,
                    over_cap=committed > self.config['cap'], jobs=jobs,
                    authority='REFERENCE_ONLY_NOT_PERMISSION')

    def _append(self, make_event):
        lock = self.root / '.writer.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            events, previous, jobs = self._state()
            event = make_event(jobs)
            event['previous'] = previous
            event['digest'] = hashlib.sha256(_canonical(event)).hexdigest()
            dest = self.root / f'event-{len(events) + 1:06d}.json'
            with dest.open('xb') as f:
                f.write(_canonical(event));f.flush();os.fsync(f.fileno())
            return self.snapshot()
        finally:
            os.close(fd)
            lock.unlink()

    def reserve(self, job, quote, *, cost_source, account_id, request_sha256):
        _identifier(job);_number(quote, 'exact current credit quote')
        if not isinstance(cost_source, str) or not cost_source.strip():
            raise ValueError('verified quote source required')
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError('verified intended account required')
        if not isinstance(request_sha256, str) or len(request_sha256) != 64 or any(c not in '0123456789abcdef' for c in request_sha256):
            raise ValueError('exact request SHA256 required')
        def event(jobs):
            if job in jobs:
                raise ValueError('job exists; recover it instead of resubmitting')
            committed = sum(j['charged'] if j['status'] == 'settled' else j['quote'] for j in jobs.values())
            if any(j['status']=='settled' and j['charged']>j['quote'] for j in jobs.values()):
                raise ValueError('unexpected overcharge: manual budget review required')
            if committed + quote > self.config['cap']:
                raise ValueError('approved credit cap would be exceeded')
            if any(j['status'] != 'settled' for j in jobs.values()):
                raise ValueError('unsettled submission: recover before new spending')
            if jobs and any(j['account_id'] != account_id for j in jobs.values()):
                raise ValueError('ledger cannot change spending account')
            return dict(kind='reserve',job=job,quote=quote,cost_source=cost_source,
                        account_id=account_id,request_sha256=request_sha256)
        return self._append(event)

    def submitted(self, job, project_id):
        _identifier(job)
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError('provider project ID required')
        def event(jobs):
            if job not in jobs or jobs[job]['status'] != 'reserved':
                raise ValueError('reservation required')
            return dict(kind='submitted',job=job,project_id=project_id)
        return self._append(event)

    def settle(self, job, charged, *, receipt_reference):
        _identifier(job);_number(charged, 'verified actual charged credits')
        if not isinstance(receipt_reference, str) or not receipt_reference.strip():
            raise ValueError('actual charge or verified non-submission evidence required')
        def event(jobs):
            if job not in jobs or jobs[job]['status'] not in {'reserved', 'submitted'}:
                raise ValueError('unsettled reservation required')
            # Record an unexpected overcharge honestly, then block later spends.
            return dict(kind='settle',job=job,charged=charged,receipt_reference=receipt_reference)
        return self._append(event)
