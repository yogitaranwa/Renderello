"""Contract tests with an explicit fake. Not model inference or photo acceptance."""
import sys,types
import pytest
from PIL import Image
from renderello.backends.klein_local import KleinEditBackend,MODEL_REVISION
from renderello.schemas import Aspect

class Tokenizer:
 def apply_chat_template(self,messages,**kw):return messages[0]['content']
 def __call__(self,text,**kw):return {'input_ids':list(range(100 if 'OVERLONG'not in text else 600))}
class FakePipe:
 def __init__(self):self.tokenizer=Tokenizer();self.kw=None
 def __call__(self,**kw):
  self.kw=kw;im=Image.new('RGB',(kw['width'],kw['height']),(1,20,50));im.putpixel((0,0),(200,100,70));return types.SimpleNamespace(images=[im])
class Backend(KleinEditBackend):
 def _load(self):return self._pipe
@pytest.fixture
def backend(monkeypatch):
 class Generator:
  def __init__(self,device):self.device=device
  def manual_seed(self,seed):self.seed=seed;return self
 import contextlib
 monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(Generator=Generator,inference_mode=contextlib.nullcontext))
 b=Backend();b._pipe=FakePipe();return b

def test_source_and_reference_are_passed(backend):
 source=Image.new('RGB',(30,40),'red');ref=Image.new('RGB',(25,25),'blue')
 out=backend.edit(source,'Do not change the label.','extra products',7,Aspect.SQUARE,ref)
 assert out.size==Aspect.SQUARE.size
 assert len(backend._pipe.kw['image'])==2
 assert backend._pipe.kw['image'][0].tobytes()==source.tobytes()
 assert backend._pipe.kw['image'][1].tobytes()==ref.tobytes()
 assert backend._pipe.kw['prompt']=='Do not change the label.\nAvoid the following: extra products'
 assert backend.last_diagnostics['revision']==MODEL_REVISION
 assert not backend.last_diagnostics['quality_accepted']
 assert not backend.last_diagnostics['magic_hour_equivalent']

@pytest.mark.parametrize('aspect',list(Aspect))
def test_aspect_full_prompt_and_four_steps(backend,aspect):
 text='Keep every small letter. '*40
 backend.edit(Image.new('RGB',(8,8)),text,'',0,aspect)
 assert backend._pipe.kw['prompt']==text
 assert backend._pipe.kw['num_inference_steps']==4
 assert backend.last_diagnostics['prompt_truncated']is False
 assert backend._pipe.kw['image'].size==(8,8)
 assert backend.last_diagnostics['negative_channel']=='none'

def test_overlong_refuses_without_generation(backend):
 with pytest.raises(ValueError,match='No truncation'):backend.edit(Image.new('RGB',(8,8)),'OVERLONG','',0,Aspect.SQUARE)
 assert backend._pipe.kw is None

@pytest.mark.parametrize('seed',[-1,True,3.2,2**63,'0'])
def test_bad_seed(backend,seed):
 with pytest.raises(ValueError):backend.edit(Image.new('RGB',(8,8)),'text','',seed,Aspect.SQUARE)

@pytest.mark.parametrize('kwargs',[{'cpu_offload':1},{'max_sequence_length':True},{'max_sequence_length':513},{'max_sequence_length':0}])
def test_bad_init(kwargs):
 with pytest.raises(ValueError):KleinEditBackend(**kwargs)

@pytest.mark.parametrize('source,prompt,negative,aspect,ref',[(None,'a','',Aspect.SQUARE,None),(Image.new('RGB',(1,1)),'','',Aspect.SQUARE,None),(Image.new('RGB',(1,1)),'a',None,Aspect.SQUARE,None),(Image.new('RGB',(1,1)),'a','','square',None),(Image.new('RGB',(1,1)),'a','',Aspect.SQUARE,'bad')])
def test_bad_inputs(backend,source,prompt,negative,aspect,ref):
 with pytest.raises(ValueError):backend.edit(source,prompt,negative,0,aspect,ref)

def test_blank_refusal(backend):
 backend._pipe.__class__=type('Blank',(FakePipe,),{'__call__':lambda self,**kw:types.SimpleNamespace(images=[Image.new('RGB',(kw['width'],kw['height']))])})
 with pytest.raises(RuntimeError,match='blank'):backend.edit(Image.new('RGB',(8,8)),'a','',0,Aspect.SQUARE)


def test_decode_dtype_guard_preserves_call_and_casts():
    from renderello.backends.klein_local import install_vae_decode_dtype_guard
    class Tensor:
        def to(self, **kw): self.cast=kw; return self
    class VAE:
        dtype='float32'
        def decode(self, sample, *args, **kwargs): return sample,args,kwargs
    vae=VAE();install_vae_decode_dtype_guard(vae);wrapped=vae.decode
    install_vae_decode_dtype_guard(vae)
    sample=Tensor();out=vae.decode(sample, 17, return_dict=False)
    assert vae.decode is wrapped
    assert sample.cast=={'dtype':'float32'}
    assert out==(sample,(17,),{'return_dict':False})
