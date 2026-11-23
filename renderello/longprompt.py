"""Long, detailed prompt support.

The owner requirement (2026-11-20): the pipeline must accept very long,
detailed prompts, not just short instructions. Two constraints shape this
module:

1. CLIP-based diffusion models (SD1.5, SDXL) tokenize with a hard context
   window (77 tokens for CLIP-L, 75 usable). Long prompts must be chunked
   and the embeddings pooled, or the tail is silently dropped.
2. API editors (Magic Hour, Qwen) take plain strings with their own limits;
   for them the full text passes through unchanged.

This module never truncates silently: chunking preserves every clause, and
`PromptPlan.report()` states exactly how the prompt will reach the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# CLIP BPE tokens roughly track word+punctuation pieces; this regex
# approximation stays within ~10% of the real tokenizer, which is enough to
# split safely below the 75-token usable window.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
CLAIP_USABLE_TOKENS = 75  # 77 minus BOS/EOS
SDXL_DUAL_ENCODER = 2     # CLIP-L + OpenCLIP-G each get the same chunks


def approx_token_count(text: str) -> int:
    """Approximate CLIP token count (overestimates slightly; safe side)."""
    return len(_TOKEN_RE.findall(text))


def _split_clauses(text: str) -> list[str]:
    """Split at sentence / clause boundaries, keeping the separators."""
    parts = re.split(r"(?<=[.!?;])\s+", text.strip())
    clauses: list[str] = []
    for part in parts:
        if approx_token_count(part) <= CLAIP_USABLE_TOKENS - 8:
            clauses.append(part)
            continue
        # Long sentence: fall back to comma boundaries, then hard word wraps.
        sub = re.split(r"(?<=,)\s+", part)
        buf = ""
        for piece in sub:
            if approx_token_count(piece) > CLAIP_USABLE_TOKENS - 8:
                words = piece.split()
                for w in words:
                    cand = f"{buf} {w}".strip()
                    if approx_token_count(cand) > CLAIP_USABLE_TOKENS - 8:
                        if buf:
                            clauses.append(buf)
                        buf = w
                    else:
                        buf = cand
            else:
                cand = f"{buf} {piece}".strip()
                if approx_token_count(cand) > CLAIP_USABLE_TOKENS - 8:
                    if buf:
                        clauses.append(buf)
                    buf = piece
                else:
                    buf = cand
        if buf:
            clauses.append(buf)
    return [c for c in clauses if c.strip()]


def chunk_prompt(text: str, max_tokens: int = CLAIP_USABLE_TOKENS - 8) -> list[str]:
    """Pack clauses into chunks that each fit one CLIP context window.

    Returns at least one chunk. No clause is dropped or truncated; a clause
    longer than the window is hard-wrapped at word boundaries.
    """
    clauses = _split_clauses(text)
    if not clauses:
        return [""]
    chunks: list[str] = []
    buf = ""
    for clause in clauses:
        cand = f"{buf} {clause}".strip()
        if approx_token_count(cand) <= max_tokens:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            buf = clause
    if buf:
        chunks.append(buf)
    return chunks


@dataclass
class PromptPlan:
    """The full resolved prompt for one generation, plus how it will be fed."""
    full_text: str
    negative: str
    chunks: list[str] = field(default_factory=list)
    provider: str = "unknown"
    approx_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "provider": self.provider,
            "approx_tokens": self.approx_tokens,
            "chunk_count": len(self.chunks),
            "chars": len(self.full_text),
            "notes": list(self.notes),
        }


def plan_prompt(full_text: str, negative: str, provider: str) -> PromptPlan:
    """Build the feeding plan for a provider.

    - diffusers/CLIP providers (sdxl_lightning, iclight): chunked at the CLIP
      window; the GPU side must embed per chunk and pool (see
      `encode_long_prompt`).
    - string providers (magichour, hf_inference, qwen, procedural): the full
      text is passed unchanged; only the leading section matters most.
    """
    plan = PromptPlan(full_text=full_text.strip(), negative=negative.strip(),
                      provider=provider)
    plan.approx_tokens = approx_token_count(plan.full_text)
    if provider in {"sdxl_lightning", "iclight", "sd15"}:
        plan.chunks = chunk_prompt(plan.full_text)
        if len(plan.chunks) > 1:
            plan.notes.append(
                f"long prompt split into {len(plan.chunks)} CLIP-window chunks; "
                "chunk embeddings are sequence-concatenated per encoder before "
                "inference (padding excluded, no clause dropped)")
    else:
        plan.chunks = [plan.full_text]
        plan.notes.append("provider consumes the prompt as a single string")
    return plan


def encode_long_prompt(pipe, prompt: str, device: str, negative: str = ""):
    """Sequence-concatenated multi-chunk text encoding for diffusers pipelines.

    Works for SDXL (two text encoders) and SD1.5 (one). Each chunk fits the
    model's context window, so no tokens are dropped; the per-chunk hidden
    states (padding excluded via the attention mask) are concatenated along
    the sequence axis, so the UNet attends over every clause - the previous
    version mean-pooled each chunk to one token and then averaged chunks,
    collapsing any prompt to a single conditioning token. Prompt and
    negative sequences are length-equalized by repeating the shorter one's
    final valid token (SDXL requires equal sequence lengths). Requires
    torch + the pipeline's own tokenizer(s); import-lazy so CPU installs
    never need torch.

    NOTE (2026-11-21): this fixes the conditioning STRUCTURE only.
    Photographic quality improvement is unproven - free-tier Colab runs
    crashed before producing comparable pixels, so no before/after visual
    validation exists yet.
    """
    import torch

    chunks = chunk_prompt(prompt)
    neg_chunks = chunk_prompt(negative) if negative else [""]

    def _encode(chunks_: list[str], tokenizer, encoder) -> "torch.Tensor":
        embs = []
        for ch in chunks_:
            tokens = tokenizer(ch, padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                out = encoder(tokens.input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-2] if hasattr(out, "hidden_states") else out[0]
            valid = int(tokens.attention_mask[0].sum().item())
            embs.append(hidden[:, :valid].float())
        return torch.cat(embs, dim=1).to(dtype=encoder.dtype)

    def _pad_to(t: "torch.Tensor", length: int) -> "torch.Tensor":
        if t.shape[1] >= length:
            return t
        pad = t[:, -1:].expand(-1, length - t.shape[1], -1)
        return torch.cat([t, pad], dim=1)

    if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2 is not None:
        pe1 = _encode(chunks, pipe.tokenizer, pipe.text_encoder)
        pe2 = _encode(chunks, pipe.tokenizer_2, pipe.text_encoder_2)
        ne1 = _encode(neg_chunks, pipe.tokenizer, pipe.text_encoder)
        ne2 = _encode(neg_chunks, pipe.tokenizer_2, pipe.text_encoder_2)
        seq = max(pe1.shape[1], ne1.shape[1])
        pe1, ne1 = _pad_to(pe1, seq), _pad_to(ne1, seq)
        pe2, ne2 = _pad_to(pe2, seq), _pad_to(ne2, seq)
        prompt_embeds = torch.cat([pe1, pe2], dim=-1)
        negative_embeds = torch.cat([ne1, ne2], dim=-1)
        pooled = pe2.mean(dim=1)
        neg_pooled = ne2.mean(dim=1)
        return prompt_embeds, negative_embeds, pooled, neg_pooled
    pe = _encode(chunks, pipe.tokenizer, pipe.text_encoder)
    ne = _encode(neg_chunks, pipe.tokenizer, pipe.text_encoder)
    seq = max(pe.shape[1], ne.shape[1])
    return _pad_to(pe, seq), _pad_to(ne, seq), None, None
