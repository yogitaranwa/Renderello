import json

import pytest
from PIL import Image

from renderello.backends.magichour import (MODEL_TABLE, MagicHourBackend,
                                           MagicHourError)
from renderello.schemas import Aspect


def test_model_table_matches_plan():
    # Plan Part B values: free-tier 640px models and creator-tier NB models.
    assert MODEL_TABLE["flux-2-klein"]["credits"] == 5
    assert MODEL_TABLE["qwen-edit"]["credits"] == 10
    assert MODEL_TABLE["qwen-edit"]["tier"] == "free"
    assert MODEL_TABLE["qwen-edit"]["max_side"] == 640
    assert MODEL_TABLE["nano-banana-2.1"]["credits"] == 50
    assert MODEL_TABLE["nano-banana-2.1"]["tier"] == "creator"
    assert MODEL_TABLE["nano-banana-pro"]["credits"] == 150


def test_mock_edit_no_network_no_credits(tmp_path):
    log = tmp_path / "cost.jsonl"
    be = MagicHourBackend(model="qwen-edit", live=False, cost_log=log)
    out = be.edit(Image.new("RGB", (300, 200), (10, 20, 30)), "p", "n", 2,
                  Aspect.SQUARE)
    assert out.size == Aspect.SQUARE.size
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows[0]["mock"] is True
    assert rows[0]["credits_charged"] == 0
    assert rows[0]["credits_listed"] == 10


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        MagicHourBackend(model="not-a-model")


def test_live_without_key_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("MAGIC_HOUR_API_KEY", raising=False)
    be = MagicHourBackend(model="qwen-edit", live=True, cost_log=tmp_path / "c.jsonl")
    with pytest.raises(MagicHourError):
        be.edit(Image.new("RGB", (64, 64)), "p", "n", 0, Aspect.SQUARE)
    with pytest.raises(MagicHourError):
        be.balance()


def test_balance_parse(monkeypatch):
    class R:
        status_code = 200
        def json(self):
            return {"id": "x", "email": "a@b.c", "tier": "creator",
                    "credits": 13806, "subscription": None}
    import requests
    monkeypatch.setenv("MAGIC_HOUR_API_KEY", "test-key")
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    info = MagicHourBackend().balance()
    assert info["credits"] == 13806 and info["tier"] == "creator"
