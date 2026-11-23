"""Frozen matched trials and read-only evidence accounting.

No generation, payment, upload, identity authentication or quality approval.
A plan is NOT execution permission. Missing first tries remain in denominators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from .image_intake import inspect_image, IntakePolicy
from .review import DIMENSIONS

SCHEMA = 'renderello.matched-trial.v1'
KINDS = {'local', 'magic_hour', 'specialist'}
STATES = {'output', 'failed', 'unconfirmed'}
TAGS = {'tiny_text', 'reflective', 'transparent', 'shoes', 'bags', 'apparel',
        'food_packaging', 'multiple_products', 'complex_edges', 'irregular_shape'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def read_json(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 8_000_000:
        raise ValueError('bounded regular JSON file required')
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def _text(value, label, limit=32000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(label + ': nonempty bounded text required')
    return value


def _identifier(value):
    _text(value, 'identifier', 80)
    if not value.isascii() or not all(c.isalnum() or c in '_-' for c in value):
        raise ValueError('identifier must be ASCII letters, digits, underscore or hyphen')
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + ': numeric value required')
    if not math.isfinite(value) or value < 0:
        raise ValueError(label + ': finite nonnegative value required')
    return value


def _relative(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('portable relative artifact path required')
    p = Path(name)
    if p.is_absolute() or any(x in ('.', '..') for x in name.split('/')):
        raise ValueError('artifact path traversal refused')
    root = Path(root).resolve()
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink artifact refused')
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError('artifact missing or outside evidence root')
    return current


def pixel_identity(image):
    """EXIF/profile-normalized RGBA samples, including size; not perceptual identity.

    Different JPEG encodings can decode differently; near duplicates still need
    curator review. Hidden transparent RGB is retained, not silently discarded.
    """
    rgba = image.convert('RGBA')
    try:
        return _sha(struct.pack('>II', *rgba.size) + rgba.tobytes())
    finally:
        rgba.close()


def freeze_plan(spec, out):
    """Validate all inputs, then save immutable normalized source snapshots.

    spec carries source paths and declared tags, not proof of photo origin or
    publication rights. A complete marker binds all files. Failed preparation
    leaves an incomplete new directory, never replaces an earlier experiment.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError('new destination required; preserve prior experiment')
    cases = spec.get('cases')
    methods = spec.get('methods')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 500:
        raise ValueError('one to 500 predeclared cases required')
    if not isinstance(methods, list) or not 2 <= len(methods) <= 12:
        raise ValueError('two to twelve methods required')
    method_ids = set()
    for method in methods:
        mid = _identifier(method['id'])
        if mid in method_ids or method.get('kind') not in KINDS:
            raise ValueError('unique methods and known method kinds required')
        method_ids.add(mid)
        if type(method.get('selection_complete')) is not bool:
            raise ValueError('explicit method-selection state required')
        if not isinstance(method.get('settings'), dict) or not method['settings']:
            raise ValueError('method settings required, including unresolved choices')
        _canonical(method)  # Refuse NaN or unserializable configuration now.
    decoded = []
    seen_ids = set()
    unique_pixels = {}
    try:
        for case in cases:
            cid = _identifier(case['id'])
            if cid in seen_ids:
                raise ValueError('duplicate case ID')
            seen_ids.add(cid)
            _text(case['template'], 'template', 80)
            _text(case['prompt'], 'prompt')
            _text(case['rights_record'], 'rights record')
            if case.get('aspect') not in {'1:1', '4:5', '3:2'}:
                raise ValueError('declared supported aspect required')
            tags = case.get('tags')
            if not isinstance(tags, list) or len(set(tags)) != len(tags) or not set(tags) <= TAGS:
                raise ValueError('unique supported category declarations required')
            if case.get('origin') not in {'photograph_claimed', 'illustration', 'unknown'}:
                raise ValueError('explicit claimed image origin required')
            source = Path(case['source'])
            if source.is_symlink():
                raise ValueError('symlink source refused')
            intake = inspect_image(source, IntakePolicy(allow_colour_conversion=True))
            digest = pixel_identity(intake.image)
            # One source may be used for different templates, but cannot become
            # two source identities or two repetitions of the same case cell.
            cell = (digest, case['template'], case['aspect'], case['prompt'])
            if cell in unique_pixels:
                intake.close()
                raise ValueError('duplicate normalized source/template/prompt cell')
            unique_pixels[cell] = cid
            # Keep metadata, not every decoded image, for a 500-case plan.
            # Reopen and reverify each input when saving its frozen snapshot.
            decoded.append((case, dict(intake.record), digest, list(intake.image.size)))
            intake.close()
        out.mkdir(parents=True)
        (out / 'inputs').mkdir()
        records = []
        for case, record, digest, size in decoded:
            name = 'inputs/' + case['id'] + '.png'
            intake = inspect_image(Path(case['source']), IntakePolicy(allow_colour_conversion=True))
            try:
                if intake.record['source_sha256'] != record['source_sha256'] or pixel_identity(intake.image) != digest:
                    raise ValueError('source changed during plan preparation')
                intake.image.save(out / name)
            finally:
                intake.close()
            records.append(dict(id=case['id'], source=name,
                                original_file_sha256=record['source_sha256'],
                                normalized_pixel_sha256=digest,
                                source_size=size,
                                intake=record,
                                template=case['template'], prompt=case['prompt'],
                                aspect=case['aspect'], tags=case['tags'],
                                origin=case['origin'], rights_record=case['rights_record']))
        plan = dict(schema=SCHEMA, state='PLAN_ONLY_NOT_EXECUTION_PERMISSION',
                    methods=methods, cases=records,
                    tasks=[dict(case=c['id'], method=m['id'], first_attempt=1)
                           for c in records for m in methods],
                    quality_status='NOT_EVALUATED',
                    selection_complete=all(m['selection_complete'] for m in methods),
                    caution='Tags, origins and rights are declarations, not independently verified facts.')
        (out / 'plan.json').write_bytes(_canonical(plan))
        hashes = {p.relative_to(out).as_posix(): _sha(p.read_bytes())
                  for p in out.rglob('*') if p.is_file()}
        (out / 'COMPLETE.json').write_bytes(_canonical(dict(schema=SCHEMA, files=hashes)))
        return plan
    finally:
        decoded.clear()


def verify_plan(root):
    """Verify exact saved snapshots and no unexpected files before accounting."""
    root = Path(root)
    complete = read_json(root / 'COMPLETE.json')
    hashes = complete.get('files')
    if complete.get('schema') != SCHEMA or not isinstance(hashes, dict) or not hashes:
        raise ValueError('invalid completion record')
    present = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if present != set(hashes) | {'COMPLETE.json'}:
        raise ValueError('unexpected or missing experiment files')
    for name, expected in hashes.items():
        if _sha(_relative(root, name).read_bytes()) != expected:
            raise ValueError('frozen experiment bytes changed: ' + name)
    plan = read_json(root / 'plan.json')
    if plan.get('schema') != SCHEMA:
        raise ValueError('unknown plan schema')
    # Rebuild original task cells instead of trusting a hand-edited task list.
    tasks = [dict(case=c['id'], method=m['id'], first_attempt=1)
             for c in plan['cases'] for m in plan['methods']]
    if tasks != plan.get('tasks'):
        raise ValueError('plan task matrix is inconsistent')
    for case in plan['cases']:
        intake = inspect_image(_relative(root, case['source']))
        try:
            if pixel_identity(intake.image) != case['normalized_pixel_sha256']:
                raise ValueError('normalized input identity mismatch')
        finally:
            intake.close()
    return plan


def _rating(record):
    if record is None:
        return None
    if not isinstance(record, dict) or set(record.get('dimensions', {})) != set(DIMENSIONS):
        raise ValueError('all seven quality dimensions required')
    if any(type(v) is not int or not 1 <= v <= 5 for v in record['dimensions'].values()):
        raise ValueError('quality ratings must be integer one to five')
    errors = record.get('critical_errors')
    if not isinstance(errors, list) or any(not isinstance(e, str) or not e.strip() for e in errors):
        raise ValueError('explicit critical-error list required')
    if type(record.get('no_critical')) is not bool or record['no_critical'] != (not errors):
        raise ValueError('critical-error declarations conflict')
    if record.get('rater') not in {'owner', 'independent_human', 'agent_eye'}:
        raise ValueError('explicit rater claim required')
    return record


def audit_results(plan_root, evidence_root, events, *, usd_per_credit=None):
    """Read-only matched accounting. Never sends, generates or infers ratings.

    Rater labels in imported files remain claims. Even complete reported ratings
    do not authenticate human identity, blinding, rights or competitive quality.
    This function consequently NEVER sets acceptance_pass=True.
    """
    plan = verify_plan(plan_root)
    if not isinstance(events, list) or len(events) > 50000:
        raise ValueError('bounded event list required')
    if usd_per_credit is not None:
        _number(usd_per_credit, 'explicit current USD per credit')
    allowed = {(t['case'], t['method']) for t in plan['tasks']}
    by_cell = {cell: {} for cell in allowed}
    output_hashes = {}
    repeated = []
    for event in events:
        cell = (event['case'], event['method'])
        if cell not in by_cell:
            raise ValueError('event is not a predeclared matched task')
        attempt = event.get('attempt')
        if type(attempt) is not int or not 1 <= attempt <= 100:
            raise ValueError('attempt must be an integer one to100')
        if attempt in by_cell[cell]:
            raise ValueError('duplicate attempt record')
        if event.get('state') not in STATES:
            raise ValueError('explicit output, failed or unconfirmed state required')
        _number(event.get('seconds'), 'measured attempt seconds')
        charge = event.get('credits_charged')
        if charge is not None:
            _number(charge, 'actual charged credits')
        if event.get('credits_estimated') is not None:
            _number(event['credits_estimated'], 'estimated credits')
        if event['state'] == 'output':
            p = _relative(evidence_root, event.get('output'))
            if _sha(p.read_bytes()) != event.get('output_sha256'):
                raise ValueError('output bytes changed')
            intake = inspect_image(p, IntakePolicy(allow_colour_conversion=True))
            try:
                identity = pixel_identity(intake.image)
            finally:
                intake.close()
            if identity in output_hashes:
                repeated.append([output_hashes[identity], [*cell, attempt]])
            output_hashes[identity] = [*cell, attempt]
        else:
            if event.get('output') or event.get('rating') is not None:
                raise ValueError('failed/unconfirmed attempt cannot carry output or quality rating')
            _text(event.get('failure'), 'explicit failed/unconfirmed reason')
        _rating(event.get('rating'))
        by_cell[cell][attempt] = event
    for attempts in by_cell.values():
        if attempts and sorted(attempts) != list(range(1, max(attempts) + 1)):
            raise ValueError('missing earlier attempt: first tries may not disappear')
    report = dict(schema=SCHEMA, quality_status='NOT_EVALUATED', acceptance_pass=None,
                  planned_tasks=len(allowed), methods={},
                  unique_normalized_sources=len({c['normalized_pixel_sha256'] for c in plan['cases']}),
                  templates=sorted({c['template'] for c in plan['cases']}),
                  declared_tags=sorted({tag for c in plan['cases'] for tag in c['tags']}),
                  missing_declared_tags=sorted(TAGS - {tag for c in plan['cases'] for tag in c['tags']}),
                  repeated_output_pixels=repeated,
                  warnings=['Declared categories and photo origins require curator verification.',
                            'Exact normalized hashes do not detect all crops or near duplicates.',
                            'Imported rater identity/blinding is not authenticated by a JSON label.',
                            'First-try failures, unconfirmed calls and missing attempts stay in planned denominators.',
                            'Successful file generation is not publication usefulness.',
                            'No model recommendation or specialist parity is inferred.'])
    for method in plan['methods']:
        cells = [by_cell[(c['id'], method['id'])] for c in plan['cases']]
        attempts = [v for cell in cells for v in cell.values()]
        first = [cell[1] for cell in cells if 1 in cell]
        outputs = [e for e in first if e['state'] == 'output']
        ratings = [e['rating'] for e in outputs if e.get('rating') is not None]
        human_claims = [r for r in ratings if r['rater'] in {'owner', 'independent_human'}]
        complete = len(human_claims) == len(outputs) and bool(first) and len(first) == len(cells)
        no_critical = sum(r['no_critical'] for r in human_claims)
        useful = sum(r['no_critical'] and r['dimensions']['publish_ready_usefulness'] >= 4
                     for r in human_claims)
        known_charges = [e['credits_charged'] for e in attempts if e.get('credits_charged') is not None]
        charged = sum(known_charges) if attempts and len(known_charges) == len(attempts) else None
        # The cost includes failed tries and retries, but an accepted-output
        # denominator requires complete claimed human ratings. It remains a
        # declared-rating calculation, never authenticated production economics.
        cost_per_useful = charged / useful if charged is not None and complete and useful else None
        n = len(cells)
        report['methods'][method['id']] = dict(
            planned_first_tries=n, recorded_first_tries=len(first),
            not_started=n - len(first), first_outputs=len(outputs),
            first_failures=sum(e['state'] == 'failed' for e in first),
            first_unconfirmed=sum(e['state'] == 'unconfirmed' for e in first),
            retries_recorded=sum(max(0, len(cell) - 1) for cell in cells),
            first_generation_rate=len(outputs) / n,
            first_attempt_median_seconds=statistics.median(e['seconds'] for e in first) if first else None,
            all_attempts=len(attempts), unknown_charge_attempts=len(attempts) - len(known_charges),
            actual_credits_total=charged, known_partial_credits=sum(known_charges),
            usd_cost_total=charged * usd_per_credit if charged is not None and usd_per_credit is not None else None,
            claimed_human_rated_first_outputs=len(human_claims),
            claimed_no_critical=no_critical if complete else None,
            claimed_publishable=useful if complete else None,
            claimed_no_critical_rate=no_critical / n if complete else None,
            claimed_publishable_rate=useful / n if complete else None,
            claimed_targets_met=(no_critical / n >= .8 and useful / n >= .7) if complete else None,
            actual_credits_per_claimed_publishable=cost_per_useful,
            rating_status='CLAIMED_HUMAN_LABELS_NOT_AUTHENTICATED' if human_claims else 'NOT_HUMAN_RATED',
            acceptance_pass=None)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='normalize snapshots; no generation or spend')
    freeze.add_argument('--spec', type=Path, required=True)
    freeze.add_argument('--out', type=Path, required=True)
    audit = sub.add_parser('audit', help='read-only audit of existing attempt records')
    audit.add_argument('--plan', type=Path, required=True)
    audit.add_argument('--evidence-root', type=Path, required=True)
    audit.add_argument('--events', type=Path, required=True)
    audit.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'freeze':
        result = freeze_plan(read_json(a.spec), a.out)
    else:
        result = audit_results(a.plan, a.evidence_root, read_json(a.events))
        with a.out.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(schema=SCHEMA, command=a.command,
                          quality_status=result['quality_status']), allow_nan=False))


if __name__ == '__main__':
    main()
