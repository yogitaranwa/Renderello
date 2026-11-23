import csv
import json
import pytest
from renderello.review import export_sheet, load_scores, DIMENSIONS


def fixture(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    output = run / 'o.png'; output.write_bytes(b'fixture bytes')
    (run / 'manifest.json').write_text(json.dumps({'source':'s.png','variants':[
        {'output':str(output),'route':'scene_compose','seed':0,'qa':{'template':'t'}}]}))
    sheet = export_sheet([run], tmp_path / 'scores.csv')
    return sheet, output


def fill(sheet, **overrides):
    with sheet.open(newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    for row in rows:
        row.update({k:'4' for k in DIMENSIONS});row.update(no_critical='true',rater='agent_eye')
        row.update(overrides)
    with sheet.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def test_blank_export_never_becomes_rating(tmp_path):
    sheet,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='complete integer'):load_scores(sheet)
    with pytest.raises(FileExistsError):export_sheet([tmp_path/'run'],sheet)


def test_seven_dimensions_and_self_reported_provenance(tmp_path):
    sheet,_=fixture(tmp_path);fill(sheet)
    result=next(iter(load_scores(sheet).values()))
    assert len(result['dimensions'])==7 and result['publishable']==4
    assert result['rater']=='agent_eye' and result['provenance_verified'] is False


def test_changed_image_rejected(tmp_path):
    sheet,output=fixture(tmp_path);fill(sheet);output.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):load_scores(sheet)


@pytest.mark.parametrize('overrides',[
    {'product_label_fidelity':'6'}, {'rater':''}, {'no_critical':'yes'},
    {'no_critical':'false','critical_errors':''},
    {'no_critical':'true','critical_errors':'changed logo'}])
def test_invalid_or_conflicting_ratings_rejected(tmp_path,overrides):
    sheet,_=fixture(tmp_path);fill(sheet,**overrides)
    with pytest.raises(ValueError):load_scores(sheet)


def test_cli_review_export_and_check(tmp_path,capsys):
    from renderello.cli import main
    sheet,_=fixture(tmp_path);fill(sheet)
    assert main(['review-check','--sheet',str(sheet)])==0
    assert 'Acceptance NOT_EVALUATED' in capsys.readouterr().out
