"""Explicit long SDXL prompt encoding with Compel's dual-CLIP contract.

No silent 77-token truncation. Chunking includes the whole text but does not
promise that a model obeys each clause or that scene quality improves.
"""


def encode_long_sdxl(pipe, prompt, negative_prompt='', *, device='cuda'):
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('nonempty full prompt required')
    if not isinstance(negative_prompt,str):raise ValueError('negative prompt must be text')
    # Bound memory explicitly and report an error instead of silently truncating.
    if len(prompt)>16000 or len(negative_prompt)>16000:
        raise ValueError('prompt exceeds explicit16000character budget; no truncation performed')
    import torch
    from compel import Compel,ReturnedEmbeddingsType
    processor=Compel(tokenizer=[pipe.tokenizer,pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder,pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False,True],truncate_long_prompts=False,device=device,
        dtype_for_device_getter=lambda _:torch.float16)
    with torch.inference_mode():
        positive,positive_pooled=processor(prompt)
        negative,negative_pooled=processor(negative_prompt)
        empty,_=processor('')
        # Explicit empty conditioning avoids the installed multi-provider's
        # missing empty_z attribute on its implicit batch-padding path.
        positive,negative=processor.pad_conditioning_tensors_to_same_length(
            [positive,negative],precomputed_padding=empty)
        conditioning=torch.cat([positive,negative],dim=0)
        pooled=torch.cat([positive_pooled,negative_pooled],dim=0)
    counts=[len(t(prompt,add_special_tokens=False)['input_ids']) for t in [pipe.tokenizer,pipe.tokenizer_2]]
    diagnostic=dict(method='Compel dual CLIP nontruncating chunks',positive_tokens_by_encoder=counts,
        conditioning_shape=list(conditioning.shape),pooled_shape=list(pooled.shape),
        truncate_long_prompts=False,full_prompt=prompt,negative_prompt=negative_prompt,
        constraint_obedience='NOT_GUARANTEED',quality_acceptance='NOT_EVALUATED')
    return dict(prompt_embeds=conditioning[0:1],negative_prompt_embeds=conditioning[1:2],
                pooled_prompt_embeds=pooled[0:1],negative_pooled_prompt_embeds=pooled[1:2]),diagnostic
