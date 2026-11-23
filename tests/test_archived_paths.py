import csv,json,shutil
import pytest
from renderello.artifact_paths import run_artifact
from renderello.review import export_sheet,load_scores,DIMENSIONS
from renderello.gallery import build_gallery


def archived(root,recorded):
    run=root/'source_clean_studio_20261110-040623_3whdgo6i';run.mkdir(parents=True)
    (run/'o.png').write_bytes(b'original image bytes')
    m={'source':'/gone/source.png','template':{'name':'Clean Studio'},'variants':[{'output':recorded.replace('RUN',run.name),'route':'scene_compose','seed':0,'qa':{'template':'clean_studio'}}]}
    (run/'manifest.json').write_text(json.dumps(m));return run


@pytest.mark.parametrize('prefix',['/kaggle/working/gpu/RUN/o.png',r'C:\Users\Owner\Renderello\RUN\o.png','o.png'])
def test_archived_outputs_rebase_only_on_exact_run(tmp_path,prefix):
    run=archived(tmp_path,prefix)
    assert run_artifact(run,prefix.replace('RUN',run.name))==run/'o.png'
    sheet=export_sheet([run],tmp_path/'scores.csv',portable=True)
    with sheet.open()as f:rows=list(csv.DictReader(f))
    assert rows[0]['output']==run.name+'/o.png'
    assert all(rows[0][k]=='' for k in DIMENSIONS)
    gallery=build_gallery([run],tmp_path/'gallery.html').read_text()
    assert run.name+'/o.png' in gallery and 'Source image not included' in gallery
    assert 'No before/after comparison is claimed' in gallery
    assert 'FAILED / NOT GENERATED' not in gallery


@pytest.mark.parametrize('recorded',['/other/o.png','../o.png','RUN/../o.png',r'C:\other\o.png','RUN/RUN/o.png','/x/NOTRUN/o.png'])
def test_no_basename_guess_or_parent_traversal(tmp_path,recorded):
    run=archived(tmp_path,'RUN/o.png')
    assert run_artifact(run,recorded.replace('RUN',run.name)) is None


def test_symlink_escape_not_resolved(tmp_path):
    run=archived(tmp_path/'runs','RUN/o.png');outside=tmp_path/'secret.png';outside.write_bytes(b'secret')
    try:
        (run/'escape.png').symlink_to(outside)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1314:
            pytest.skip('Windows account lacks symlink creation privilege; escape check remains covered on Linux')
        raise
    assert run_artifact(run,'escape.png')is None


def test_portable_ratings_move_and_hash_check_survive(tmp_path):
    old=tmp_path/'old';run=archived(old,'/gone/RUN/o.png');sheet=export_sheet([run],old/'scores.csv',portable=True)
    with sheet.open()as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:row.update({k:'4'for k in DIMENSIONS});row.update(rater='agent_eye',no_critical='true')
    with sheet.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    new=tmp_path/'new';shutil.move(old,new);scores=load_scores(new/'scores.csv')
    assert str(new/run.name/'o.png')in scores
    assert next(iter(scores.values()))['provenance_verified']is False
    (new/run.name/'o.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):load_scores(new/'scores.csv')


def test_source_override_requires_exact_content_hash(tmp_path):
    import hashlib
    run=archived(tmp_path,'/gone/RUN/o.png');source=tmp_path/'photo.jpg';source.write_bytes(b'known source')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();mp=run/'manifest.json';m=json.loads(mp.read_text());m['source_provenance']={'sha256':digest};mp.write_text(json.dumps(m))
    text=build_gallery([run],tmp_path/'gallery.html',source_paths={digest:source}).read_text();assert 'src="photo.jpg"'in text
    source.write_bytes(b'wrong photo')
    with pytest.raises(ValueError,match='hash mismatch'):build_gallery([run],tmp_path/'bad.html',source_paths={digest:source})
