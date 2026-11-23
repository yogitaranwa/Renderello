"""Regression tests for longprompt.encode_long_prompt conditioning structure.

The 2026-11-20 defect: _encode mean-pooled every chunk to one token and then
averaged across chunks, collapsing ANY prompt (short or long) to a single
(1, 1, dim) conditioning token - the UNet could not attend to the prompt's
clauses, and long prompts also diverged in sequence length from the negative
(4 chunks vs 1), which SDXL rejects.

These tests use a numpy-backed fake torch + fake tokenizer/encoder (no GPU,
no model weights) and prove the STRUCTURE of the fix only: full sequence
length preserved, padding excluded, prompt/negative lengths equalized.
They say nothing about photographic output quality.
"""
import sys
import types

import numpy as np
import pytest

from renderello import longprompt


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    @property
    def shape(self):
        return self.arr.shape

    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def mean(self, dim=None, keepdim=False):
        return FakeTensor(self.arr.mean(axis=dim, keepdims=keepdim))

    def sum(self):
        return FakeTensor(self.arr.sum())

    def item(self):
        return float(self.arr)

    def expand(self, *dims):
        arr = self.arr
        for axis, d in enumerate(dims):
            if d != -1:
                arr = np.repeat(arr, d, axis=axis)
        return FakeTensor(arr)

    def __getitem__(self, idx):
        return FakeTensor(self.arr[idx])


def _cat(tensors, dim=0):
    return FakeTensor(np.concatenate([t.arr for t in tensors], axis=dim))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def fake_torch(monkeypatch):
    mod = types.ModuleType("torch")
    mod.cat = _cat
    mod.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class FakeBatch:
    def __init__(self, n_valid, max_length):
        self.input_ids = FakeTensor(np.zeros((1, max_length)))
        mask = np.zeros((1, max_length), dtype=np.float32)
        mask[0, :n_valid] = 1.0
        self.attention_mask = FakeTensor(mask)

    def to(self, device):
        return self


class FakeTokenizer:
    """Whitespace tokenizer: valid length = word count + BOS/EOS, right-padded."""

    def __init__(self, max_length=12):
        self.model_max_length = max_length

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        n = min(len(text.split()) + 2, max_length)
        return FakeBatch(n, max_length)


class FakeEncoderOut:
    def __init__(self, hidden):
        self.hidden_states = [None, FakeTensor(hidden), None]


class FakeEncoder:
    """Hidden state position (b, i, d) = i + d/1000; dim 4."""

    dtype = "float32"

    def __call__(self, input_ids, output_hidden_states=True):
        seq = input_ids.arr.shape[1]
        hidden = np.zeros((1, seq, 4), dtype=np.float32)
        for i in range(seq):
            hidden[0, i, :] = i + np.arange(4) / 1000.0
        return FakeEncoderOut(hidden)


class FakePipe:
    def __init__(self, max_length=12):
        self.tokenizer = FakeTokenizer(max_length)
        self.tokenizer_2 = FakeTokenizer(max_length)
        self.text_encoder = FakeEncoder()
        self.text_encoder_2 = FakeEncoder()


def _long_prompt():
    # Force several chunks: each word ~1 approx token, window 67.
    return " ".join(f"clause{i} " + "word " * 20 for i in range(6))


def test_long_prompt_sequence_not_collapsed(fake_torch):
    pipe = FakePipe()
    prompt = _long_prompt()
    chunks = longprompt.chunk_prompt(prompt)
    assert len(chunks) > 1, "test requires a genuinely multi-chunk prompt"
    pe, ne, pooled, neg_pooled = longprompt.encode_long_prompt(
        pipe, prompt, "cpu", "")
    expected = sum(min(len(c.split()) + 2, 12) for c in chunks)
    assert pe.shape[1] == expected, (
        f"conditioning collapsed: seq {pe.shape[1]} != {expected} valid tokens")
    assert pe.shape[1] > 1


def test_prompt_and_negative_sequence_lengths_equalized(fake_torch):
    pipe = FakePipe()
    pe, ne, _, _ = longprompt.encode_long_prompt(pipe, _long_prompt(), "cpu", "")
    assert pe.shape[1] == ne.shape[1]
    assert pe.shape == ne.shape


def test_negative_longer_than_prompt_also_equalized(fake_torch):
    pipe = FakePipe()
    pe, ne, _, _ = longprompt.encode_long_prompt(
        pipe, "short prompt", "cpu", _long_prompt())
    assert pe.shape[1] == ne.shape[1]


def test_padding_tokens_excluded(fake_torch):
    pipe = FakePipe(max_length=12)
    pe, ne, _, _ = longprompt.encode_long_prompt(
        pipe, "three word prompt", "cpu", "")
    # 3 words + BOS/EOS = 5 valid, not the padded 12.
    assert pe.shape[1] == 5


def test_padding_repeat_is_final_valid_token(fake_torch):
    pipe = FakePipe()
    _, ne, _, _ = longprompt.encode_long_prompt(
        pipe, _long_prompt(), "cpu", "")
    # negative "" -> 1 chunk -> BOS+EOS = 2 valid tokens, index 1 = 1.0
    last = ne.arr[0, -1, 0]
    assert last == pytest.approx(1.0), "pad region must repeat the final valid token"


def test_every_chunk_contributes_in_order(fake_torch):
    pipe = FakePipe()
    prompt = _long_prompt()
    chunks = longprompt.chunk_prompt(prompt)
    pe, _, _, _ = longprompt.encode_long_prompt(pipe, prompt, "cpu", "")
    # Position values restart at 0 for each chunk; the concat must contain
    # each chunk's local positions in order: count occurrences of position 0.
    zeros = int((np.isclose(pe.arr[0, :, 0], 0.0)).sum())
    assert zeros == len(chunks), "each chunk's first token must survive"


def test_single_encoder_pipeline_shape(fake_torch):
    pipe = FakePipe()
    pipe.tokenizer_2 = None
    pe, ne, pooled, neg_pooled = longprompt.encode_long_prompt(
        pipe, _long_prompt(), "cpu", "")
    assert pe.shape[1] == ne.shape[1] > 1
    assert pooled is None and neg_pooled is None


def test_pooled_embeddings_present_for_sdxl(fake_torch):
    pipe = FakePipe()
    _, _, pooled, neg_pooled = longprompt.encode_long_prompt(
        pipe, _long_prompt(), "cpu", "")
    assert pooled is not None and pooled.shape[-1] == 4
    assert neg_pooled is not None


def test_chunk_prompt_still_drops_nothing():
    prompt = _long_prompt()
    chunks = longprompt.chunk_prompt(prompt)
    rejoined = " ".join(chunks).split()
    assert rejoined == prompt.split()
