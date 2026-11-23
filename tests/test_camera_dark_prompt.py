import numpy as np
from PIL import Image
from renderello.iclight_gpu import CameraMatchedCallback

def test_charcoal_not_negatively_conditioned_as_black():
    class Model:
        def relight(self,canvas,prompt,**kw):
            assert 'charcoal' in prompt
            assert 'black background' not in kw['n_prompt']
            return [Image.fromarray(canvas)]
    CameraMatchedCallback(Model(),style='luxury_hero')(np.full((100,100,3),176,np.uint8),'ignored')
