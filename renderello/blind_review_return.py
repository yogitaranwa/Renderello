"""Import a separate returned score sheet without modifying the sealed pack.

No reviewer identity, independent review, product acceptance or winner is inferred.
Reports contain the private method key and must not be sent back to blind raters.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
from .blind_review import verify_pack
from .brief_acceptance import Rating, evaluate
from .review import DIMENSIONS


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError('bounded regular file required')
    return path.read_bytes()


def import_return(pack, returned_csv, *, provenance, review_reference):
    """Return hash-bound judgments and explicit missing-review entries.

    provenance is supplied by the caller from trusted context, never selected by
    a rater's text in a CSV. claimed_independent records a claim, not verification.
    All tokens must appear exactly once, including failed generations. Blank or
    partial rows stay incomplete; invalid values fail closed, never become zero.
    Critical errors use a JSON string array, including explicit [] for none.
    """
    if provenance not in ('assistant_judgment', 'owner', 'claimed_independent'):
        raise ValueError('explicit review provenance required')
    if not isinstance(review_reference, str) or not review_reference.strip() or len(review_reference) > 200:
        raise ValueError('bounded review reference required')
    pack, returned_csv = Path(pack), Path(returned_csv)
    sealed = verify_pack(pack)
    if sealed.get('schema_version') != 2:
        raise ValueError('a schema-2 pack with a sealed private key is required; legacy packs remain unchanged')
    raw = _read(returned_csv)
    if returned_csv.resolve().is_relative_to(pack.resolve()):
        raise ValueError('returned score sheet must be separate from sealed pack')
    original = list(csv.DictReader(io.StringIO(_read(pack/'REVIEW/scores.csv').decode('utf-8-sig'))))
    key = json.loads(_read(pack/'PRIVATE/key.json'))
    candidates = key['candidates']
    lookup = {c['token']: c for c in candidates}
    expected = {r['token']: r for r in original}
    if len(lookup) != len(candidates) or set(lookup) != set(expected):
        raise ValueError('private key does not match sealed score tokens')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    headers = list(original[0])
    if reader.fieldnames != headers:
        raise ValueError('returned score columns/order must match sealed sheet exactly')
    rows = list(reader)
    if len(rows) != len(expected) or len({r.get('token') for r in rows}) != len(rows) or {r.get('token') for r in rows} != set(expected):
        raise ValueError('each planned token must appear exactly once, including failures')
    results=[]
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('malformed score row')
        token = row['token']; c = lookup[token]
        if row['status'] != expected[token]['status']:
            raise ValueError('generation status is sealed and cannot be changed by returned ratings')
        out = dict(token=token, method=c['method'], original_output_sha256=c['original_sha256'],
                   generation_status=row['status'], failure=c['failure'],
                   rater_text_unverified=row['rater'], notes=row['notes'], rating=None)
        provided = {d: row[d].strip() for d in DIMENSIONS}
        for v in provided.values():
            if v and v not in ('1','2','3','4','5'):
                raise ValueError('scores must be blank or integer 1..5')
        decision = row['no_critical'].strip().lower()
        if decision not in ('','true','false'):
            raise ValueError('no_critical must be blank, true or false')
        error_text = row['critical_errors'].strip()
        errors = None if not error_text else json.loads(error_text)
        if errors is not None and (not isinstance(errors,list) or any(not isinstance(x,str) or not x.strip() for x in errors)):
            raise ValueError('critical_errors must be a JSON string array')
        if decision and errors is not None and (decision == 'true') == bool(errors):
            raise ValueError('critical decision conflicts with errors')
        any_judgment = any(provided.values()) or decision or error_text
        if row['status'] == 'NOT_GENERATED':
            if any_judgment: raise ValueError('failed generation cannot receive image scores')
            out['review_status']='NOT_GENERATED_RETAINED_IN_DENOMINATOR'
        elif not all(provided.values()) or not decision or errors is None or not row['rater'].strip():
            out['review_status']='INCOMPLETE_REVIEW'
            out['missing_fields']=[d for d,v in provided.items() if not v]
            out['missing_fields'] += [name for name,ok in [('no_critical',bool(decision)),('critical_errors',errors is not None),('rater',bool(row['rater'].strip()))] if not ok]
        else:
            rating=Rating(token,c['original_sha256'],{d:int(v) for d,v in provided.items()},decision=='true',tuple(errors),provenance,review_reference+':'+token)
            out['review_status']='COMPLETE_RECORDED_JUDGMENT'
            out['rating']=dict(output_id=rating.output_id,output_sha256=rating.output_sha256,scores=dict(rating.scores),no_critical=rating.no_critical,critical_errors=list(rating.critical_errors),provenance=rating.provenance,review_reference=rating.review_reference)
        results.append(out)
    # Recheck sealed content after parsing. A hash proves consistency, not origin.
    verify_pack(pack)
    return dict(schema='blind-review-return.v1',returned_csv_sha256=hashlib.sha256(raw).hexdigest(),
                private_key_sha256=sealed['private_key_sha256'],source_sha256=key['source_sha256'],review_reference=review_reference,
                provenance=provenance,denominator=len(candidates),
                complete_reviews=sum(r['rating'] is not None for r in results),results=results,
                identity_semantics='caller_provenance_and_rater_text_not_identity_authentication',
                disclosure='PRIVATE_METHOD_KEY_INCLUDED_KEEP_AWAY_FROM_BLIND_RATERS',
                best_quality_model=None,value_fallback=None,production_go=False,live_calls=0)



def bind_ratings(return_report, plan, attempts, token_output_ids):
    """Bind selected returned tokens to one method's preregistered attempt plan.

    An explicit map avoids guessing template/alternative from a filename. This
    validates consistency with the supplied report, not the report's authorship.
    Call import_return on the sealed local pack first. Never reassign a judgment
    to another native image, source or method, even when scores look plausible.
    """
    evaluate(plan, attempts, [])
    if not isinstance(token_output_ids, dict) or not token_output_ids:
        raise ValueError('explicit nonempty token-to-output mapping required')
    if len(set(token_output_ids.values())) != len(token_output_ids):
        raise ValueError('one returned judgment cannot pad multiple planned slots')
    if return_report.get('schema') != 'blind-review-return.v1':
        raise ValueError('returned-review report required')
    rows={r['token']:r for r in return_report['results']}
    by_id={p.output_id:p for p in plan}; result=[]
    for token, output_id in token_output_ids.items():
        if token not in rows or output_id not in by_id:
            raise ValueError('unknown review token or unplanned output')
        row=rows[token]; planned=by_id[output_id]
        if planned.method != row['method'] or planned.source_sha256 != return_report['source_sha256']:
            raise ValueError('returned judgment source/method does not match plan')
        candidate_attempts=[a for a in attempts if a.output_id==output_id]
        if row['rating'] is None:
            if row['generation_status']=='NOT_GENERATED' and any(a.status=='success' for a in candidate_attempts):
                raise ValueError('failed review entry conflicts with successful attempt')
            continue
        r=row['rating']
        if r['output_id'] != token or r['output_sha256'] != row['original_output_sha256']:
            raise ValueError('returned judgment does not bind the native candidate')
        result.append(Rating(output_id,r['output_sha256'],r['scores'],r['no_critical'],tuple(r['critical_errors']),r['provenance'],r['review_reference']))
    # Existing evaluator enforces original hash/first-try/no-substituted-retry rules.
    evaluate(plan, attempts, result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--returned-csv',type=Path,required=True)
    p.add_argument('--provenance',required=True,choices=['assistant_judgment','owner','claimed_independent'])
    p.add_argument('--review-reference',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.output.resolve().is_relative_to(a.pack.resolve()):
            raise ValueError('output must be outside sealed pack')
        result=import_return(a.pack,a.returned_csv,provenance=a.provenance,review_reference=a.review_reference)
        with a.output.open('x') as f:json.dump(result,f,indent=2)
    except (ValueError,TypeError,KeyError,OSError,UnicodeError,csv.Error) as exc:
        p.exit(2,'REFUSED: '+str(exc)+'\n')
    print('Saved private returned-review report: '+str(a.output))


if __name__=='__main__':main()
