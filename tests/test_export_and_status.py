import json
from pathlib import Path
import numpy as np
from PIL import Image
from renderello import export
from renderello.benchmark import _summarize


def test_export_roundtrip_exact(tmp_path):
    arr = np.random.default_rng(1).integers(0, 255, (64, 48, 3), dtype=np.uint8)
    p = tmp_path / "x.png"
    Image.fromarray(arr).save(p)
    assert export.verify_roundtrip(p)
    assert "sha256=" in export.manifest_line(p)


def _rows(n=4):
    return [{"output": f"o{i}", "template": "t", "route": "r", "critical_errors": [],
             "score_fidelity": 5, "seconds": 1.0, "success": True,
             "credits_estimated": 0} for i in range(n)]


def test_quality_gates_not_evaluated_without_human_scores():
    s = _summarize(_rows(), 4)
    assert s["quality_status"] == "NOT_EVALUATED"
    for k in ("no_critical_alteration", "publish_ready"):
        assert s["gates"][k]["pass"] is None
        assert s["gates"][k]["status"] == "NOT_EVALUATED"
        assert s["gates"][k]["automated_proxy_pass"] is True


def test_quality_gates_from_complete_human_scores():
    h = {f"o{i}": {"no_critical": i != 0, "publishable": 4 if i else 2, "rater": "owner"} for i in range(4)}
    s = _summarize(_rows(), 4, human=h)
    assert s["quality_status"] == "HUMAN_EVALUATED"
    assert s["gates"]["no_critical_alteration"]["pass"] is False  # 3/4 = 0.75 < 0.80
    h2 = {k: {"no_critical": True, "publishable": 5, "rater": "independent_human"} for k in h}
    s2 = _summarize(_rows(), 4, human=h2)
    assert s2["gates"]["publish_ready"]["pass"] is True


def test_incomplete_human_scores_stay_not_evaluated():
    s = _summarize(_rows(), 4, human={"o0": {"no_critical": True, "publishable": 5}})
    assert s["quality_status"] == "NOT_EVALUATED"


def test_agent_eye_review_is_not_human_evaluation():
    h = {f"o{i}": {"no_critical": True, "publishable": 5, "rater": "agent_eye"} for i in range(4)}
    s = _summarize(_rows(), 4, human=h)
    assert s["quality_status"] == "AGENT_EYE_REVIEWED"
    assert s["gates"]["publish_ready"]["pass"] is None
    h2 = {k: {"no_critical": True, "publishable": 5} for k in h}  # no rater recorded
    assert _summarize(_rows(), 4, human=h2)["quality_status"] == "NOT_EVALUATED"
    h3 = dict(h); h3["o0"] = {"no_critical": True, "publishable": 5, "rater": "owner"}
    assert _summarize(_rows(), 4, human=h3)["quality_status"] == "NOT_EVALUATED"  # mixed


def test_nested_route_proxy_cannot_become_quality_recommendation():
    s = _summarize(_rows(), 4)
    assert s["routes"]["r"]["gate_80pct_no_critical"] is None
    assert s["routes"]["r"]["automated_proxy_80pct_no_critical"] is True
    assert s["config_comparison"]["best_quality_route"] is None
    assert s["config_comparison"]["fallback_route"] is None
    assert s["config_comparison"]["recommendation_status"] == "NOT_EVALUATED"


def test_twenty_variants_are_not_twenty_unique_sources():
    rows = [dict(_rows(1)[0], output=f"o{i}", source="one.png", template=f"t{i%8}") for i in range(20)]
    human = {r["output"]: {"no_critical": True, "publishable": 5, "rater": "owner"} for r in rows}
    s = _summarize(rows, 20, human=human)
    assert s["quality_status"] == "HUMAN_EVALUATED"  # per-sample rating provenance only
    assert s["coverage"]["unique_sources"] == 1
    assert s["coverage"]["source_count_met"] is False
    assert s["coverage"]["template_count_met"] is True
    assert s["acceptance_pass"] is None
    assert s["specialist_comparison_status"] == "NOT_EVALUATED"
    for i, row in enumerate(rows):
        row["source"] = f"s{i}.png"
    s2 = _summarize(rows, 20, human=human)
    assert s2["coverage"]["source_count_met"] is False  # paths without content identity
    for i, row in enumerate(rows):
        row["source_sha256"] = f"{i:064x}"
    s3 = _summarize(rows, 20, human=human)
    assert s3["coverage"]["source_count_met"] is True
    assert s3["coverage"]["real_photo_count_met"] is None
    assert s3["acceptance_pass"] is None  # no specialist evidence


def test_roundtrip_avoids_deprecated_pixel_api(tmp_path, monkeypatch):
    def deprecated(*args, **kwargs):
        raise AssertionError('deprecated getdata must not be called')
    monkeypatch.setattr(Image.Image, 'getdata', deprecated)
    for mode in ('RGB', 'RGBA', 'L', 'P'):
        p = tmp_path / (mode + '.png')
        img = Image.new(mode, (17, 23))
        if mode == 'RGBA':
            img.putpixel((8, 9), (13, 27, 95, 123))
        img.save(p)
        assert export.verify_roundtrip(p)
