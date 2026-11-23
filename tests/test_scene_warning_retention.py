"""A scene's own paste diagnostic must not erase independent QA evidence."""
import json
import numpy as np
from PIL import Image
from renderello import pipeline, qa
from renderello.config import RunConfig
from renderello.schemas import EditRequest, Route, QAReport
from renderello.templates import get_template


def test_scene_keeps_independent_pixel_warning_even_when_paste_claims_exact(tmp_path,monkeypatch):
    rgba=np.zeros((70,60,4),np.uint8);rgba[:,:,:3]=(30,100,160);rgba[10:60,15:45,3]=255
    source=tmp_path/'source.png';Image.fromarray(rgba).save(source)
    def independent_warning(*args,**kw):
        return QAReport(str(source),'output','scene_compose','clean_studio',0,
            ssim_product=1,color_drift=0,edge_retained=1,
            critical_errors=['independent detail-change warning'],score_fidelity=2)
    monkeypatch.setattr(qa,'evaluate',independent_warning)
    monkeypatch.setattr(qa,'matte_sanity',lambda *a,**kw:['independent matte warning'])
    run=pipeline.run_pipeline(EditRequest(source,get_template('clean_studio')),
        [Route.SCENE_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'),
        scene_fn=lambda image,prompt:np.full_like(image,200),use_embedded_alpha=True)
    v=run.variants[0];m=json.loads(run.manifest_path.read_text())
    assert v.success and 'paste_bit_exact=True' in v.qa.notes
    assert v.qa.critical_errors==['independent detail-change warning','independent matte warning']
    assert v.qa.score_fidelity==2
    assert m['variants'][0]['qa']['critical_errors']==v.qa.critical_errors
