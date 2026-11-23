import json
from pathlib import Path
from renderello.source_provenance import identify_source
from renderello.gallery import _source_caption


def test_registry_content_match_survives_rename(tmp_path):
    src=Path(__file__).parents[1]/'benchmark/sources/watch1.jpg'
    copy=tmp_path/'arbitrary-upload.jpg';copy.write_bytes(src.read_bytes())
    a=identify_source(copy)
    assert a['match_basis']=='exact_file_sha256'
    assert 'gadgetdude' in a['attribution']
    assert 'current legal clearance' in a['rights_status']


def test_named_but_changed_source_is_unknown(tmp_path):
    p=tmp_path/'watch1.jpg';p.write_bytes(b'not the registered photo')
    a=identify_source(p)
    assert a['match_basis']=='no_registered_content_match'
    assert 'UNKNOWN' in a['classification']


def test_illustration_distinct_from_product_photo():
    p=Path(__file__).parents[1]/'benchmark/sources/tshirt1.jpg'
    assert 'illustration' in identify_source(p)['classification']


def test_registry_bytes_all_match_pack():
    root=Path(__file__).parents[1]
    records=json.loads((root/'renderello/source-registry.json').read_text())
    sources=json.loads((root/'benchmark/manifest.json').read_text())['sources']
    assert len(records)==27
    assert len({r['sha256'] for r in records})==27
    for label in ['glass-full','watch-full','deodorant-backlabel']:
        candidate=root/'benchmark/sources'/f'{label}.jpg'
        metadata=identify_source(candidate)
        assert metadata['match_basis']=='exact_file_sha256'
        assert 'commons.wikimedia.org/wiki/File:' in metadata['attribution']
    for s in sources:
        assert identify_source(root/'benchmark'/s['path'])['attribution']==s['license']


def test_caption_visible_rights_and_legacy_unknown():
    assert 'UNKNOWN' in _source_caption({})
    p=Path(__file__).parents[1]/'benchmark/sources/watch1.jpg'
    caption=_source_caption({'source_provenance':identify_source(p)})
    assert 'gadgetdude' in caption and 'flickr.com/photos/' in caption
    assert 'SHA256' in caption and 'Private test use' in caption


def test_gallery_exposes_record_and_escapes_text(tmp_path):
    from renderello.gallery import build_gallery
    run=tmp_path/'run';run.mkdir()
    source=tmp_path/'source.jpg';source.write_bytes(b'test')
    (run/'manifest.json').write_text(json.dumps({'source':str(source),'source_provenance':{'classification':'External <image>','attribution':'creator & URL','sha256':'abc','rights_status':'pending'},'template':{'name':'Studio'},'variants':[]}))
    page=build_gallery([run],tmp_path/'index.html').read_text()
    assert 'External &lt;image&gt;' in page and 'creator &amp; URL' in page
    assert 'Sources are synthetic test fixtures' not in page


def test_benchmark_deduplicates_content_and_exposes_classes():
    from renderello.benchmark import _summarize
    rows=[]
    for i in range(25):
        rows.append({'source':f'copy-{i}.jpg','source_sha256':'same_content_hash','source_classification':'External illustration','template':'studio','route':'cutout_compose','critical_errors':[],'score_fidelity':5,'output':f'o{i}.png'})
    c=_summarize(rows,25)['coverage']
    assert c['unique_sources']==25 and c['unique_content_sources']==1
    assert c['source_classifications']=={'External illustration':1}
    assert c['source_count_met'] is False and c['real_photo_count_met'] is None


def test_confirmed_cgi_snack_not_physical_photo_even_renamed(tmp_path):
    src=Path(__file__).parents[1]/'benchmark/sources/snack1.jpg'
    p=tmp_path/'actual_food_photo.jpg';p.write_bytes(src.read_bytes())
    r=identify_source(p)
    assert r['capture_origin']=='SYNTHETIC_CONFIRMED_BY_OWNING_PAGE_DESCRIPTION'
    assert r['physical_photo_eligible'] is False
    assert 'Cinema4D' in r['origin_evidence']
    assert 'CGI' in r['classification']
    assert r['origin_evidence_url']=='https://www.flickr.com/photos/42042252@N02/5559242063'

def test_unknown_source_does_not_become_verified_photo(tmp_path):
    p=tmp_path/'photo.jpg';p.write_bytes(b'unknown')
    r=identify_source(p)
    assert r['physical_photo_eligible'] is None
    assert r['capture_origin']=='UNVERIFIED'
