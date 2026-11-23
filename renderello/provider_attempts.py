"""Read recorded pipeline provider evidence into existing typed economics inputs.

Explicit preregistered output mapping is required. This authenticates neither
provider responses, account denomination, reviewer identity nor current prices.
"""
from pathlib import Path
import hashlib,json,argparse,math
from dataclasses import asdict
from .brief_acceptance import PlannedOutput,Attempt,evaluate


def import_attempts(plan, runs):
    """runs is [(manifest_path, {integer_variant_index: planned_output_id}), ...].

    List runs in actual chronological attempt order. No retry is inferred from
    filenames. Repeated project IDs are rejected to avoid charging recovery twice.
    Failed/partial artifacts remain failed; unknown provider outcome stays unknown.
    Legacy variants without per-call evidence are refused rather than priced at0.
    """
    plan=list(plan);evaluate(plan,[],[]);planned={p.output_id:p for p in plan}
    attempts=[];seen_ids=set();seen_projects=set();references=[]
    for manifest_path,mapping in runs:
        path=Path(manifest_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size>5_000_000:
            raise ValueError('bounded regular pipeline manifest required')
        raw=path.read_bytes();m=json.loads(raw)
        if not isinstance(mapping,dict) or not mapping or any(type(k)is not int for k in mapping):
            raise ValueError('explicit integer variant-index mapping required')
        recorded={i for i,v in enumerate(m['variants']) if isinstance(v.get('provider_evidence'),dict)}
        if recorded and set(mapping)!=recorded:
            raise ValueError('map every recorded provider attempt, including failures; no cherry-picked billing denominator')
        source_hash=m.get('image_intake',{}).get('source_sha256')
        for i,output_id in mapping.items():
            if output_id not in planned or not 0<=i<len(m['variants']):raise ValueError('unknown planned output or variant index')
            p=planned[output_id];v=m['variants'][i];e=v.get('provider_evidence')
            if not isinstance(e,dict) or e.get('schema')!='provider-attempt.v1':raise ValueError('legacy/missing per-call provider evidence is not a zero-cost run')
            if p.source_sha256 is None or p.source_sha256!=source_hash or p.template_id!=m['template']['key'] or p.method!=e.get('model'):
                raise ValueError('preregistered source/template/model mismatch')
            aid=e.get('attempt_id');project=e.get('project_id')
            if not isinstance(aid,str) or not aid or aid in seen_ids:raise ValueError('duplicate/missing provider attempt ID')
            seen_ids.add(aid)
            if project:
                if project in seen_projects:raise ValueError('repeated provider project: recovery cannot be counted as a new charge')
                seen_projects.add(project)
            credit=e.get('credits_charged')
            if credit is not None and (type(credit) not in (int,float) or not math.isfinite(credit) or credit<0):raise ValueError('invalid recorded charge')
            terminal=e.get('terminal_detail_observed') is True and e.get('provider_status') in ('complete','error','canceled')
            if not terminal and credit is not None:raise ValueError('nonterminal charge cannot be treated as settled')
            complete=v.get('success') is True and e.get('pipeline_completed') is True
            digest=None
            if complete:
                image=Path(v['output']);digest=e.get('output_sha256')
                if not image.is_absolute():raise ValueError('absolute recorded artifact path required; relocate through explicit evidence recovery first')
                if image.is_symlink() or not image.is_file() or image.stat().st_size>40_000_000 or hashlib.sha256(image.read_bytes()).hexdigest()!=digest:
                    raise ValueError('successful artifact bytes are missing or changed')
                if e.get('provider_status')!='complete':raise ValueError('successful artifact conflicts with provider outcome')
            status='success' if complete else 'failed' if terminal else 'unknown'
            attempts.append(Attempt(output_id,aid,status,v.get('seconds'),credit,digest,'' if complete else 'Recorded pipeline failure/partial artifact' if terminal else 'Provider outcome unresolved; do not resubmit automatically'))
            references.append(dict(attempt_id=aid,manifest=str(path),manifest_sha256=hashlib.sha256(raw).hexdigest(),variant_index=i,project_id=project,mock=e.get('mock'),receipt_path=e.get('receipt_path')))
    evaluate(plan,attempts,[])
    return attempts,references


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--recover-stage',type=Path,help='explicitly recover this retained staging directory to --output without overwriting')
    mode.add_argument('--validate-output',action='store_true',help='verify --output and references against their completion marker')
    mode.add_argument('--input',type=Path,help='JSON with explicit plan, ordered runs (manifest, mapping), as_of; optional charge_credit_system declaration')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        from .evidence_pair import export_pair,recover_pair,validate_pair
        if a.validate_output:
            commit=validate_pair(a.output);print('Verified complete payload/reference pair hashes. Persistence: '+commit.get('persistence','legacy_unspecified')+'; no power-loss guarantee.');return
        if a.recover_stage:
            commit=recover_pair(a.recover_stage,a.output);print('Recovered and verified complete pair; staging retained. Original persistence: '+commit.get('persistence','legacy_unspecified')+'; no power-loss guarantee.');return
        if a.input.is_symlink() or not a.input.is_file() or a.input.stat().st_size>2_000_000:raise ValueError('bounded input required')
        request=json.loads(a.input.read_text())
        if set(request)-{'plan','runs','as_of','charge_credit_system'}:raise ValueError('unknown import fields')
        plan=[PlannedOutput(**v) for v in request['plan']]
        runs=[]
        for r in request['runs']:
            if set(r)!={'manifest','mapping'}:raise ValueError('exact run manifest/mapping fields required')
            pairs=r['mapping']
            if not isinstance(pairs,list) or any(set(x)!={'variant_index','output_id'} for x in pairs):raise ValueError('explicit mapping pairs required')
            mapping={x['variant_index']:x['output_id'] for x in pairs}
            if len(mapping)!=len(pairs):raise ValueError('duplicate mapped variant')
            runs.append((a.input.resolve().parent/Path(r['manifest']),mapping))
        attempts,refs=import_attempts(plan,runs)
        result=dict(plan=[asdict(v) for v in plan],attempts=[asdict(v) for v in attempts],ratings=[],as_of=request['as_of'])
        if 'charge_credit_system'in request:result['charge_credit_system']=request['charge_credit_system']
        # Validate through existing strict adapter before saving compatible payload.
        from .brief_economics import from_payload
        from_payload(result)
        result['evidence_pair_required']=True
        stage=export_pair(a.output,result,dict(records=refs,semantics='recorded_evidence_not_provider_or_account_authentication; ratings_and_price_not_inferred'))
    except (ValueError,TypeError,KeyError,OSError) as exc:p.exit(2,'REFUSED: '+str(exc)+'\n')
    print('Saved and hash-verified complete payload/reference pair; staging retained at '+str(stage)+'; no ratings/price/winner inferred. Persistence: '+validate_pair(a.output)['persistence']+'; no power-loss guarantee.')


if __name__=='__main__':main()
