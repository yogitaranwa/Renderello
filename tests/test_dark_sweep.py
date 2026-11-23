import numpy as np
from renderello.iclight_gpu import CameraMatchedCallback,DarkSweepConditioningCallback

def test_conditioning_only_ablation_does_not_change_prompt():
    class Recorder:
        def __init__(self):self.calls=[]
        def relight(self,canvas,prompt,**kw):self.calls.append((canvas,prompt,kw));return canvas,0.
    m=Recorder();c=np.full((640,512,3),176,np.uint8)
    CameraMatchedCallback(m,style='luxury_hero')(c,'unused')
    DarkSweepConditioningCallback(m)(c,'unused')
    old,new=m.calls
    assert old[1:]==new[1:]
    assert np.all(old[0]==176) and 18<=new[0].min()<new[0].max()<60
    assert new[0].shape==c.shape and new[0].dtype==np.uint8
    assert np.array_equal(new[0][...,0],new[0][...,2])
    assert np.all(c==176)
