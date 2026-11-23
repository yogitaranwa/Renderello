"""Tests for the v1.0 modules: longprompt, presets, multinput, studio."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from renderello.longprompt import chunk_prompt, plan_prompt
from renderello.presets import PRESETS, build_detailed_prompt, presets_for_ui
from renderello.multinput import (compose_group, cutout_products,
                                  extract_reference_style)
from renderello.schemas import Aspect
from renderello.studio import StudioRequest, run_studio


def _square(color, size=(200, 180)):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[50:150, 40:140] = color
    return Image.fromarray(a)


def test_chunk_prompt_splits_long_text():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_prompt(text, max_tokens=77)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 77


def test_chunk_prompt_short_text_single_chunk():
    assert chunk_prompt("short prompt") == ["short prompt"]


def test_plan_prompt_magichour_untruncated():
    text = " ".join(f"detail{i}" for i in range(300))
    plan = plan_prompt(text, "neg", "magichour")
    assert "detail299" in plan.full_text  # no 77-token truncation


def test_presets_all_have_researched_prompts():
    assert len(PRESETS) >= 8  # brief requires 8+ templates
    for p in PRESETS.values():
        assert len(p.detailed_prompt) > 200, p.key
        assert p.template.category and p.tagline


def test_presets_for_ui_shape():
    ui = presets_for_ui()
    assert len(ui) == len(PRESETS)
    for row in ui:
        assert {"key", "name", "tagline"} <= set(row)


def test_build_detailed_prompt_includes_material_and_instruction():
    p = build_detailed_prompt(PRESETS["clean_studio"], instruction="on wet slate",
                              material="glass", product_description="amber bottle")
    assert "wet slate" in p and "amber bottle" in p
    assert "glass" in p.lower()


def test_reference_style_extraction(tmp_path):
    f = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (200, 120, 40)).save(f)
    style = extract_reference_style(f)
    assert style.dominant_rgb[0] > 150
    assert 0.0 <= style.brightness <= 1.0
    assert "Style reference" in style.prompt_fragment()


def test_cutout_and_compose_group(tmp_path):
    paths = []
    for i, c in enumerate(((30, 144, 200), (220, 120, 30))):
        p = tmp_path / f"p{i}.png"
        _square(c).save(p)
        paths.append(p)
    members = cutout_products(paths, tmp_path / "cuts")
    assert all(m.error is None for m in members)
    preset = PRESETS["flat_lay"]
    arr = compose_group(members, preset.template, preset.template.aspect, 7)
    assert arr.ndim == 3 and arr.shape[2] == 3


def test_studio_compose_and_manifest(tmp_path):
    p = tmp_path / "p.png"
    _square((30, 144, 200)).save(p)
    res = run_studio(StudioRequest(product_paths=[p], preset_key="clean_studio",
                                   variants=1, provider="procedural"),
                     tmp_path / "runs")
    assert res.variants[0].error is None
    m = json.loads((res.run_dir / "manifest.json").read_text())
    assert m["mode"] == "cutout-compose"
    assert m["variants"][0]["qa"]["native_pixel_audit"]["status"] == "MATCH"
    assert "may resize" in m["honesty"]
    assert (res.run_dir / "contact_sheet.png").exists()


def test_studio_magichour_unconfigured_refuses_zero_bill(tmp_path, monkeypatch):
    monkeypatch.delenv("RENDERELLO_MH_LIVE", raising=False)
    monkeypatch.delenv("RENDERELLO_MH_DEV_MOCK", raising=False)
    monkeypatch.delenv("MAGIC_HOUR_API_KEY", raising=False)
    p = tmp_path / "p.png"
    _square((30, 144, 200)).save(p)
    with pytest.raises(RuntimeError, match="zero credits"):
        run_studio(StudioRequest(product_paths=[p], preset_key="luxury_hero",
                                 variants=1, provider="magichour"),
                   tmp_path / "runs")


def test_studio_magichour_dev_mock_is_labeled(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    p = tmp_path / "p.png"
    _square((30, 144, 200)).save(p)
    res = run_studio(StudioRequest(product_paths=[p], preset_key="clean_studio",
                                   variants=1, provider="magichour"),
                     tmp_path / "runs")
    assert res.variants[0].error is None
    m = json.loads((res.run_dir / "manifest.json").read_text())
    assert m["live"] is False
    assert "PLACEHOLDER" in m["credits_note"]
    assert "zero credits" in m["credits_note"]


def test_mh_budget_guard_unset_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_LIVE", "1")
    monkeypatch.setenv("MAGIC_HOUR_API_KEY", "dummy")
    monkeypatch.delenv("RENDERELLO_MH_BUDGET_CREDITS", raising=False)
    from renderello.studio import _enforce_budget
    with pytest.raises(RuntimeError, match="OFF"):
        _enforce_budget("qwen-edit")


def test_mh_budget_guard_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "led.jsonl"))
    (tmp_path / "led.jsonl").write_text(
        '{"credits_spent": 42}\n{"credits_spent": 3}\n')
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "50")
    from renderello.studio import _enforce_budget
    _enforce_budget("flux-2-klein")   # 45 + 5 = 50, fits exactly
    with pytest.raises(RuntimeError, match="exhausted"):
        _enforce_budget("qwen-edit")  # 45 + 10 > 50 -> refuse


def test_mh_budget_guard_within_cap_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "led.jsonl"))
    (tmp_path / "led.jsonl").write_text('{"credits_spent": 30}\n')
    monkeypatch.setenv("RENDERELLO_MH_BUDGET_CREDITS", "50")
    from renderello.studio import _enforce_budget, record_spend
    _enforce_budget("qwen-edit")  # 30 + 10 <= 50 ok
    record_spend("qwen-edit", 10, 10)
    _enforce_budget("qwen-edit")  # 40 + 10 = 50, fits exactly
    record_spend("qwen-edit", 10, 10)  # now 50 spent
    with pytest.raises(RuntimeError, match="exhausted"):
        _enforce_budget("flux-2-klein")  # 50 + 5 > 50 -> refuse


def test_record_spend_unknown_counts_listed(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "led.jsonl"))
    from renderello.studio import record_spend
    record_spend("qwen-edit", None, 10)  # uncertain -> conservative listed
    row = json.loads((tmp_path / "led.jsonl").read_text().strip())
    assert row["credits_spent"] == 10
    assert row["charged_observed"] is False


def test_cutout_auto_fallback_warns_and_records(tmp_path, monkeypatch):
    """Silent BiRefNet->grabcut fallback caused an unverifiable hosted run;
    the fallback must warn and the result must name the real backend."""
    import sys
    import types
    from renderello import cutout as cutout_mod
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    # Force a torch-present, onnxruntime-absent host so the auto chain is
    # exactly birefnet -> grabcut regardless of the local environment.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    def _boom(img, device="cpu"):
        raise ImportError("No module named 'kornia'")

    monkeypatch.setitem(cutout_mod._BACKENDS, "birefnet", _boom)
    with pytest.warns(RuntimeWarning, match="falling back"):
        res = cutout_mod.cutout(_square((30, 144, 200)), tmp_path, backend="auto")
    assert res.backend == "grabcut"


def test_studio_manifest_records_cutout_backend(tmp_path):
    p = tmp_path / "p.png"
    _square((30, 144, 200)).save(p)
    res = run_studio(StudioRequest(product_paths=[p], preset_key="clean_studio",
                                   variants=1, provider="procedural"),
                     tmp_path / "runs")
    m = json.loads((res.run_dir / "manifest.json").read_text())
    prod = m["products"][0]
    assert prod["cutout_backend"] in ("grabcut", "birefnet", "u2netp_onnx")
    assert 0.0 < prod["cutout_coverage"] <= 1.0
    if prod["cutout_backend"] == "grabcut":
        assert any("grabcut" in w for w in m["warnings"])
    else:
        assert not any("grabcut" in w for w in m["warnings"])


def test_unsupported_luxury_mock_refuses_before_backend(tmp_path, monkeypatch):
    import pytest
    from renderello.empty_scene_prompt import UnsupportedEmptySceneStaging
    from renderello.backends.magichour import MagicHourBackend
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    p=tmp_path/"p.png";_square((30,144,200)).save(p)
    def forbidden(*args,**kwargs):raise AssertionError("backend reached for unsupported staging")
    monkeypatch.setattr(MagicHourBackend,"__init__",forbidden)
    with pytest.raises(UnsupportedEmptySceneStaging):
        run_studio(StudioRequest(product_paths=[p],preset_key="luxury_hero",variants=1,provider="magichour"),tmp_path/"runs")
