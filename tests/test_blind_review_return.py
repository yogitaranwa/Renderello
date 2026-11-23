from pathlib import Path
import csv,json,sys,subprocess
import pytest
from PIL import Image
from renderello.blind_review import Candidate,build_pack,verify_pack
from renderello.blind_review_return import import_return
from renderello.review import DIMENSIONS


def fixture(tmp):
    paths=[]
    for i,color in enumerate(['red','green','blue']):
        p=tmp/f'p{i}.png';Image.new('RGB',(16,16),color).save(p);paths.append(p)
    pack=build_pack(paths[0],[Candidate('m1',paths[1]),Candidate('m2',paths[2]),Candidate('failed',None,'no output')],tmp/'pack',seed=5)
    rows=list(csv.DictReader((pack/'REVIEW/scores.csv').open()))
    returned=tmp/'returned.csv'
    return pack,returned,rows


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def score(row):
    row.update({d:'4' for d in DIMENSIONS});row.update(no_critical='true',critical_errors='[]',rater='Claimed Rater')


def load(pack,path,**kw):return import_return(pack,path,provenance='claimed_independent',review_reference='test-return',**kw)


def test_blank_retains_all_tokens_not_fake_ratings(tmp_path):
    pack,path,rows=fixture(tmp_path);write(path,rows);r=load(pack,path)
    assert r['denominator']==3 and r['complete_reviews']==0
    assert sum(x['review_status']=='INCOMPLETE_REVIEW' for x in r['results'])==2
    assert not r['production_go'] and r['best_quality_model'] is None
    verify_pack(pack)


def test_complete_hash_bound_scores_no_reviewer_authentication(tmp_path):
    pack,path,rows=fixture(tmp_path)
    for row in rows:
        if row['status']=='OUTPUT_PRESENT':score(row)
    write(path,rows);r=load(pack,path)
    assert r['complete_reviews']==2 and r['denominator']==3
    key={x['token']:x for x in json.loads((pack/'PRIVATE/key.json').read_text())['candidates']}
    for x in r['results']:
        assert x['method']==key[x['token']]['method']
        if x['rating']:assert x['rating']['output_sha256']==key[x['token']]['original_sha256']
    assert 'not_identity_authentication' in r['identity_semantics']
    assert r['provenance']=='claimed_independent'


def test_partial_scores_stay_unknown(tmp_path):
    pack,path,rows=fixture(tmp_path);row=next(x for x in rows if x['status']=='OUTPUT_PRESENT');score(row);row[DIMENSIONS[0]]=''
    write(path,rows);r=load(pack,path);assert r['complete_reviews']==0


@pytest.mark.parametrize('change',['duplicate','missing','unknown','status','badscore','float','critical','errors','failedscore','missingcolumn','extracolumn'])
def test_invalid_return_refused(tmp_path,change):
    pack,path,rows=fixture(tmp_path);row=next(x for x in rows if x['status']=='OUTPUT_PRESENT');score(row)
    if change=='duplicate':rows[1]=dict(rows[0])
    elif change=='missing':rows.pop()
    elif change=='unknown':row['token']='C99'
    elif change=='status':row['status']='NOT_GENERATED'
    elif change=='badscore':row[DIMENSIONS[0]]='0'
    elif change=='float':row[DIMENSIONS[0]]='4.0'
    elif change=='critical':row['critical_errors']='["changed logo"]'
    elif change=='errors':row['critical_errors']='{}'
    elif change=='failedscore':score(next(x for x in rows if x['status']=='NOT_GENERATED'))
    elif change=='missingcolumn':
        for x in rows:x.pop('notes')
    elif change=='extracolumn':
        for x in rows:x['identity_verified']='true'
    write(path,rows)
    with pytest.raises(ValueError):load(pack,path)


def test_tampered_private_key_rejected(tmp_path):
    pack,path,rows=fixture(tmp_path);write(path,rows);key=pack/'PRIVATE/key.json';x=json.loads(key.read_text());x['candidates'][0]['method']='swapped';key.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='key'):load(pack,path)


def test_legacy_manifest_refused_without_rewriting(tmp_path):
    pack,path,rows=fixture(tmp_path);write(path,rows);m=pack/'COMPLETE.json';x=json.loads(m.read_text());x['schema_version']=1;x.pop('private_key_sha256');m.write_text(json.dumps(x));before=m.read_bytes()
    with pytest.raises(ValueError,match='schema-2'):load(pack,path)
    assert m.read_bytes()==before


def test_csv_provenance_never_selects_identity(tmp_path):
    pack,path,rows=fixture(tmp_path)
    for row in rows:
        if row['status']=='OUTPUT_PRESENT':score(row);row['rater']='verified owner'
    write(path,rows);r=load(pack,path)
    assert all(x['rating']['provenance']=='claimed_independent' for x in r['results'] if x['rating'])


def test_cli_exclusive_output_and_pack_unchanged(tmp_path):
    pack,path,rows=fixture(tmp_path);write(path,rows);out=tmp_path/'report.json'
    cmd=[sys.executable,'-m','renderello.blind_review_return','--pack',str(pack),'--returned-csv',str(path),'--provenance','assistant_judgment','--review-reference','test','--output',str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    before=out.read_bytes();assert subprocess.run(cmd,capture_output=True).returncode==2;assert out.read_bytes()==before
    verify_pack(pack)


@pytest.mark.parametrize('kind',['modified_image','edited_sealed_csv','separate_symlink','malformed_csv','changed_decision','missing_rater','explicit_critical'])
def test_additional_integrity_and_incomplete_cases(tmp_path,kind):
    pack,path,rows=fixture(tmp_path);row=next(x for x in rows if x['status']=='OUTPUT_PRESENT');score(row)
    if kind=='changed_decision':row['no_critical']='maybe'
    if kind=='missing_rater':row['rater']=''
    if kind=='explicit_critical':row.update(no_critical='false',critical_errors='["invented label"]')
    write(path,rows)
    if kind=='modified_image':(pack/'REVIEW'/f"{row['token']}.png").write_bytes(b'wrong')
    if kind=='edited_sealed_csv':(pack/'REVIEW/scores.csv').write_text('edited')
    if kind=='separate_symlink':
        linked=tmp_path/'link.csv';linked.symlink_to(path);path=linked
    if kind=='malformed_csv':path.write_text(path.read_text()+',,,,,,,,,,,,,,,,,,\n')
    if kind in ('missing_rater','explicit_critical'):
        r=load(pack,path)
        if kind=='missing_rater':assert r['complete_reviews']==0
        else:
            rated=next(x['rating'] for x in r['results'] if x['rating']);assert not rated['no_critical'] and rated['critical_errors']==['invented label']
    else:
        with pytest.raises(ValueError):load(pack,path)


def test_bind_to_preregistered_source_method_native_hash(tmp_path):
    from renderello.blind_review_return import bind_ratings
    from renderello.brief_acceptance import PlannedOutput,Attempt
    pack,path,rows=fixture(tmp_path)
    for row in rows:
        if row['status']=='OUTPUT_PRESENT':score(row)
    write(path,rows);r=load(pack,path);row=next(x for x in r['results'] if x['rating'])
    p=[PlannedOutput('real-output','source','template',row['method'],0,source_sha256=r['source_sha256'])]
    a=[Attempt('real-output','real-attempt','success',None,50,row['original_output_sha256'])]
    ratings=bind_ratings(r,p,a,{row['token']:'real-output'})
    assert len(ratings)==1 and ratings[0].output_id=='real-output'
    from dataclasses import replace
    for wrong in [replace(p[0],method='wrong'),replace(p[0],source_sha256='f'*64)]:
        with pytest.raises(ValueError):bind_ratings(r,[wrong],a,{row['token']:'real-output'})
    with pytest.raises(ValueError):bind_ratings(r,p,[replace(a[0],output_sha256='e'*64)],{row['token']:'real-output'})
