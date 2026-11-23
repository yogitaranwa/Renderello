import json
from pathlib import Path

from renderello.benchmark import run_benchmark
from renderello.backends.base import StubEditor
from renderello.config import RunConfig
from tests.conftest import make_product


def test_benchmark_gates_have_numerators(tmp_path):
    srcs = tmp_path / "sources"
    srcs.mkdir()
    for i in range(3):
        make_product().save(srcs / f"p{i}.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "sources": [{"path": f"sources/p{i}.png", "label": f"synthetic {i}",
                     "license": "synthetic test fixture"} for i in range(3)],
        "templates": ["clean_studio", "luxury_hero"],
        "routes": ["cutout_compose", "pure_prompt"],
        "variants": 1}))
    cfg = RunConfig(seed=0, variants=1, out_root=tmp_path / "runs")
    csv_path = run_benchmark(manifest, cfg, editor=StubEditor())
    assert csv_path.exists()
    report = json.loads((csv_path.parent / "benchmark_report.json").read_text())
    gates = report["gates"]
    assert gates["no_critical_alteration"]["denominator"] == 12  # 3 src x 2 tpl x 2 routes
    assert gates["no_critical_alteration"]["target"] == 0.80
    assert gates["publish_ready"]["target"] == 0.70
    assert gates["templates_represented"]["denominator"] == 8
    assert gates["templates_represented"]["pass"] is False  # only 2 templates
    assert "human_review_queue" in gates
    assert report["coverage"]["unique_sources"] == 3
    assert report["coverage"]["unique_content_sources"] == 1  # identical fixture, 3 paths
    assert report["coverage"]["source_count_met"] is False
    assert report["coverage"]["real_photo_count_met"] is None
    assert "UNKNOWN" in next(iter(report["coverage"]["source_classifications"]))
    assert report["config_comparison"]["best_quality_route"] is None
    assert report["config_comparison"]["fallback_route"] is None
    assert report["config_comparison"]["automated_fidelity_leader"] in (
        "cutout_compose", "pure_prompt")
    assert "NOT EVALUATED on real data" in report["gate_status"]
    # PDF economics: median latency, cost per successful output, retries.
    for route, stats in report["routes"].items():
        assert stats["median_seconds"] is not None and stats["median_seconds"] >= 0
        assert stats["success_rate"] is not None
        assert "retries_needed" in stats and "credits_total" in stats
        if stats["credits_total"] == 0:  # local routes: no per-output credit cost
            assert stats["credits_per_successful_output"] is None


def test_manifest_requires_license(tmp_path):
    import pytest
    (tmp_path / "s.png").write_bytes(make_product().tobytes() or b"x")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sources": [{"path": "s.png", "label": "x"}]}))
    with pytest.raises((ValueError, FileNotFoundError)):
        from renderello.benchmark import load_manifest
        load_manifest(bad)


def test_summarize_runs_reuses_manifests(tmp_path):
    """Gate math must aggregate existing runs (post-QA-change rescoring, GPU
    rerun avoidance) with the same numbers as a fresh benchmark."""
    from renderello.benchmark import summarize_runs
    rd = tmp_path / "run1"
    rd.mkdir()
    (rd / "o.png").write_bytes(b"\x89PNG")
    (rd / "manifest.json").write_text(json.dumps({
        "source": "s.png", "template": {"key": "clean_studio"},
        "variants": [
            {"route": "cutout_compose", "seed": 1, "output": str(rd / "o.png"),
             "backend": "grabcut", "seconds": 1.5, "credits_estimated": 0,
             "success": True,
             "qa": {"template": "clean_studio", "score_fidelity": 5,
                    "critical_errors": []}},
            {"route": "cutout_compose", "seed": 2, "output": str(rd / "missing.png"),
             "backend": "grabcut", "seconds": 0.5, "credits_estimated": 0,
             "success": False,
             "qa": {"template": "clean_studio", "score_fidelity": 1,
                    "critical_errors": ["variant failed: RuntimeError: x"]}}]}))
    csv_path = summarize_runs([rd], tmp_path / "sum",
                              source_meta={"s.png": {"label": "l", "license": "CC0"}})
    rep = json.loads((csv_path.parent / "benchmark_report.json").read_text())
    assert rep["variants"] == 2
    assert rep["coverage"]["unique_content_sources"] == 0
    assert rep["coverage"]["source_identity_complete"] is False
    assert rep["gates"]["no_critical_alteration"]["numerator"] == 1
    st = rep["routes"]["cutout_compose"]
    assert st["median_seconds"] == 1.0
    assert st["retries_needed"] == 1
    assert rep["gates"]["human_review_queue"] == []
    assert rep["gates"]["execution_failure_queue"][0]["errors"]
