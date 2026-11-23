"""Coverage checks for all draft templates. Passing is not visual acceptance."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello.templates import TEMPLATES
from renderello.template_locked_preview import preview_reviewed

@pytest.mark.parametrize('template', list(TEMPLATES))
def test_every_template_preserves_native_reviewed_core(tmp_path, template):
    pixels=np.zeros((40,32,3),dtype=np.uint8)
    yy,xx=np.mgrid[:40,:32]
    pixels[:]=np.stack([xx*7,yy*5,(xx+yy)*3],axis=-1)
    alpha=np.zeros((40,32),dtype=np.uint8);alpha[4:36,5:27]=255
    source=tmp_path/'source.png';mask=tmp_path/'alpha.png'
    Image.fromarray(pixels).save(source);Image.fromarray(alpha).save(mask)
    instruction='Preserve every symbol. '+('Do not replace the product texture or label. '*100)
    result=preview_reviewed(source,mask,{'core':[7,7,25,33]},tmp_path/'result',
        reviewed=True,template=template,instruction=instruction,alternatives=2,
        material='opaque',offset=[496,492])
    report=json.loads(Path(result['manifest']).read_text())
    assert report['custom_instruction_recorded_not_generatively_executed']==instruction
    assert report['paid_credits']==0 and len(report['outputs'])==2
    for path,audit in zip(report['outputs'],report['audits']):
        image=Image.open(path)
        assert image.size==(1024,1024) and image.mode=='RGB'
        assert np.array_equal(np.asarray(image)[499:525,503:521],pixels[7:33,7:25])
        assert audit['opaque_changed_pixels']==0
        assert audit['source_resampled'] is False
        assert audit['source_rgb_repainted'] is False
        assert audit['protected_regions'][0]['changed_pixels']==0
