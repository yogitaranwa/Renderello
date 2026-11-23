import json
from pathlib import Path
from renderello.gallery import build_gallery, _rel


def test_gallery_paths_resolve_across_sibling_folders_and_warn_fallback(tmp_path):
    from PIL import Image
    run=tmp_path/'runs'/'one';run.mkdir(parents=True)
    src=run/'source.png';out=run/'out.png'
    Image.new('RGB',(20,20),'red').save(src);Image.new('RGB',(20,20),'blue').save(out)
    (run/'manifest.json').write_text(json.dumps({'source':str(src),'template':{'name':'Studio'},'variants':[
       {'route':'cutout_compose','seed':0,'output':str(out),'backend':'grabcut','qa':{'score_fidelity':5,'critical_errors':[]}}]}))
    page=build_gallery([run],tmp_path/'gallery'/'index.html')
    assert (page.parent/_rel(str(out),page.parent)).resolve()==out
    assert 'REVIEW REQUIRED: GrabCut fallback' in page.read_text()
    assert 'meta good' not in page.read_text()


def test_gallery_cutout_failed_legacy_manifest_is_visible_not_crash(tmp_path):
    run=tmp_path/'failed';run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'source':'flat.png','template':{'name':'Studio'},'cutout_error':'empty cutout','variants':[{'route':'cutout_compose','seed':0,'qa':{'score_fidelity':1,'critical_errors':['cutout failed: <empty>']}}]}))
    page=build_gallery([run],tmp_path/'gallery.html').read_text()
    assert 'FAILED / NOT GENERATED' in page and 'No image output and no quality rating' in page
    assert 'cutout failed: &lt;empty&gt;' in page
    assert 'auto fidelity 1/5' not in page


def test_gallery_missing_output_file_has_no_broken_output_image(tmp_path):
    run=tmp_path/'missing';run.mkdir();missing=tmp_path/'missing.png'
    (run/'manifest.json').write_text(json.dumps({'source':'source.png','template':{'name':'Studio'},'variants':[{'route':'scene_compose','seed':0,'output':str(missing),'qa':{}}]}))
    page=build_gallery([run],tmp_path/'gallery.html').read_text()
    assert 'FAILED / NOT GENERATED' in page and 'Output file is missing' in page
    assert 'src="missing.png"' not in page
