"""Opt-in FLUX.2[klein]4B source editor. Never selected as the default.

Owning model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
Apache2 per card, pinned revision. This is NOT the MagicHour API configuration.
Quality and device fit need an actual run. Model edits can change labels/geometry.
No exact-product, physical-relight or publication-acceptance guarantee.
"""
from __future__ import annotations
import threading
from PIL import Image
from ..schemas import Aspect

MODEL_ID='black-forest-labs/FLUX.2-klein-4B'
MODEL_REVISION='e7b7dc27f91deacad38e78976d1f2b499d76a294'

def install_vae_decode_dtype_guard(vae):
    """Match decode sample to VAE weights; preserve original decode/hooks.

    Flux2KleinPipeline casts decode latents to device but not VAE dtype.
    T4 fp16 denoising with a float32 VAE therefore needs this boundary cast.
    This does not alter prompts, seeds, denoising or model weights.
    """
    if getattr(vae, '_renderello_decode_dtype_guard', False):
        return
    original_decode = vae.decode
    def decode_with_matching_dtype(sample, *args, **kwargs):
        return original_decode(sample.to(dtype=vae.dtype), *args, **kwargs)
    vae.decode = decode_with_matching_dtype
    vae._renderello_decode_dtype_guard = True

class KleinEditBackend:
    name='flux2-klein-4b-local-experimental'
    _lock=threading.RLock()

    def __init__(self, *, cpu_offload: bool=True, max_sequence_length: int=512):
        if type(cpu_offload) is not bool:raise ValueError('cpu_offload must be boolean')
        if type(max_sequence_length) is not int or not 32<=max_sequence_length<=512:
            raise ValueError('max_sequence_length must be integer32..512')
        self.cpu_offload=cpu_offload
        self.max_sequence_length=max_sequence_length
        self._pipe=None
        self.last_diagnostics={}

    def _load(self):
        import torch
        from diffusers import Flux2KleinPipeline
        if not torch.cuda.is_available():raise RuntimeError('Klein editor requires CUDA here; no CPU fallback silently selected')
        if self._pipe is None:
            major,_=torch.cuda.get_device_capability()
            dtype=torch.bfloat16 if major>=8 else torch.float16
            pipe=Flux2KleinPipeline.from_pretrained(MODEL_ID,revision=MODEL_REVISION,torch_dtype=dtype)
            # VAE's owning config force_upcast=True. T4 lacks native bf16.
            pipe.vae.to(dtype=torch.float32)
            install_vae_decode_dtype_guard(pipe.vae)
            if self.cpu_offload:pipe.enable_model_cpu_offload()
            else:pipe.to('cuda')
            pipe.vae.enable_slicing()
            self._pipe=pipe
        return self._pipe

    def edit(self, source: Image.Image, prompt: str, negative_prompt: str,
             seed: int, aspect: Aspect, reference: Image.Image|None=None)->Image.Image:
        if not isinstance(source,Image.Image):raise ValueError('PIL source required')
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('nonempty prompt required')
        if not isinstance(negative_prompt,str):raise ValueError('negative_prompt must be string')
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('seed must be integer0..2**63-1')
        if not isinstance(aspect,Aspect):raise ValueError('known Aspect required')
        if reference is not None and not isinstance(reference,Image.Image):raise ValueError('PIL reference required')
        # Step-distilled Klein has no separate negative-prompt channel. Do not
        # silently drop exclusions: preserve them as explicit narrative text.
        text=prompt if not negative_prompt.strip() else prompt+'\nAvoid the following: '+negative_prompt
        with self._lock:
            pipe=self._load()
            rendered=pipe.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=pipe.tokenizer(rendered,truncation=False)['input_ids']
            if len(ids)>self.max_sequence_length:
                raise ValueError(f'Full prompt requires{len(ids)}tokens; limit{self.max_sequence_length}. No truncation performed.')
            images=source.convert('RGB') if reference is None else [source.convert('RGB'),reference.convert('RGB')]
            import torch
            w,h=aspect.size
            self.last_diagnostics=dict(model=MODEL_ID,revision=MODEL_REVISION,source_conditioned=True,reference_count=1 if reference is None else 2,full_prompt=text,prompt_tokens=len(ids),prompt_truncated=False,steps=4,negative_channel='narrative_append' if negative_prompt.strip() else 'none',magic_hour_equivalent=False,quality_accepted=False)
            with torch.inference_mode():
                out=pipe(image=images,prompt=text,width=w,height=h,num_inference_steps=4,guidance_scale=1.,max_sequence_length=self.max_sequence_length,generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
            if not isinstance(out,Image.Image) or out.size!=(w,h):raise RuntimeError('Editor returned wrong image type/dimensions')
            if out.mode!='RGB':out=out.convert('RGB')
            if all(high-low<=2 for low,high in out.getextrema()):raise RuntimeError('Editor returned blank image')
            return out

    def close(self):
        with self._lock:
            if self._pipe is not None:
                self._pipe=None
                import torch,gc
                gc.collect();torch.cuda.empty_cache()
