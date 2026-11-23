import json

from renderello.backends.base import StubEditor
from renderello.config import RunConfig
from renderello.pipeline import run_pipeline
from renderello.schemas import EditRequest, Route
from renderello.templates import get_template


def _cfg(tmp_path):
    return RunConfig(seed=11, variants=2, out_root=tmp_path / "runs")


def test_all_three_routes_smoke(product_path, tmp_path):
    req = EditRequest(source_path=product_path,
                      template=get_template("clean_studio"), seed=11)
    res = run_pipeline(req, [Route.CUTOUT_COMPOSE, Route.PURE_PROMPT,
                             Route.HYBRID], _cfg(tmp_path), editor=StubEditor())
    assert len(res.variants) == 6  # 3 routes x 2 variants
    ok = [v for v in res.variants if v.output_path.exists()]
    assert len(ok) == 6
    manifest = json.loads(res.manifest_path.read_text())
    assert manifest["template"]["key"] == "clean_studio"
    assert len(manifest["variants"]) == 6
    for v in manifest["variants"]:
        assert v["qa"]["score_fidelity"] >= 1


def test_route2_only_needs_no_editor(product_path, tmp_path):
    req = EditRequest(source_path=product_path,
                      template=get_template("luxury_hero"), seed=3)
    res = run_pipeline(req, [Route.CUTOUT_COMPOSE], _cfg(tmp_path), editor=None)
    assert all(v.route is Route.CUTOUT_COMPOSE for v in res.variants)
    assert all(v.qa.critical_errors == [] for v in res.variants)


def test_backend_failure_recorded_not_hidden(product_path, tmp_path):
    class BoomEditor(StubEditor):
        name = "boom"
        def edit(self, *a, **k):
            raise RuntimeError("no gpu here")
    req = EditRequest(source_path=product_path,
                      template=get_template("social_ad"), seed=0)
    res = run_pipeline(req, [Route.PURE_PROMPT], _cfg(tmp_path), editor=BoomEditor())
    assert res.variants[0].qa.score_fidelity is None
    assert not res.variants[0].success and not res.variants[0].output_path.exists()
    assert res.best_by_qa() is None
    assert "variant failed" in res.variants[0].qa.critical_errors[0]
    assert "no gpu here" in res.variants[0].qa.critical_errors[0]


def test_bad_instruction_blocks_run(product_path, tmp_path):
    import pytest
    req = EditRequest(source_path=product_path,
                      template=get_template("clean_studio"),
                      custom_instruction="remove the text on the label")
    with pytest.raises(ValueError):
        run_pipeline(req, [Route.CUTOUT_COMPOSE], _cfg(tmp_path))


def test_variant_seed_bounds(product_path, tmp_path):
    cfg = RunConfig(seed=5, variants=9, out_root=tmp_path)  # clamps to 4
    assert cfg.variant_seeds() == [5, 6, 7, 8]


def test_scene_compose_route_with_fake_scene(product_path, tmp_path):
    import numpy as np
    from renderello.pipeline import run_pipeline
    from renderello.schemas import Route
    from renderello.config import RunConfig
    from renderello.templates import get_template
    from renderello.schemas import EditRequest
    cfg = RunConfig(out_root=tmp_path)
    fake = lambda gray, prompt: np.full_like(gray, (30, 90, 160))
    req = EditRequest(source_path=product_path, template=get_template("luxury_hero"))
    res = run_pipeline(req, [Route.SCENE_COMPOSE], cfg, scene_fn=fake)
    v = res.variants[0]
    assert v.output_path.exists() and not any("variant failed" in e for e in v.qa.critical_errors)
    res2 = run_pipeline(req, [Route.SCENE_COMPOSE], cfg)   # no scene_fn: recorded failure, not a crash
    assert any("needs a scene_fn" in e for e in res2.variants[0].qa.critical_errors)


def test_cli_scene_compose_mock(tmp_path, product_path):
    from renderello.cli import main
    rc = main(["run", "--source", str(product_path), "--templates", "luxury_hero", "--routes", "scene_compose",
               "--scene", "mock", "--variants", "1", "--out", str(tmp_path / "o")])
    assert rc == 0 and list((tmp_path / "o").rglob("scene_compose_seed*.png"))


def test_scene_compose_glass_mode_on_clean_backdrop_and_refuses_clutter(tmp_path):
    import numpy as np
    from PIL import Image
    from renderello.pipeline import run_pipeline
    from renderello.schemas import Route, EditRequest
    from renderello.config import RunConfig
    from renderello.templates import get_template
    a = np.full((400, 300, 3), 250, np.uint8)
    a[60:340, 100:200] = 235; a[60:340, 100:106] = 90; a[60:340, 194:200] = 90
    a[60:66, 100:200] = 90; a[334:340, 100:200] = 90
    p = tmp_path / "glass.png"; Image.fromarray(a).save(p)
    fake = lambda gray, prompt: np.full_like(gray, (20, 120, 40))
    res = run_pipeline(EditRequest(source_path=p, template=get_template("clean_studio")),
                       [Route.SCENE_COMPOSE], RunConfig(out_root=tmp_path / "r", variants=1),
                       scene_fn=fake, glass_mode=True)
    v = res.variants[0]
    assert v.output_path.exists()
    assert "GLASS MODE" in v.qa.notes
    assert "REVIEW NEEDED (glass mode)" in v.qa.notes
    assert v.qa.score_fidelity is None  # never invent a score
    assert res.best_by_qa() is None or res.best_by_qa() is not v
    assert not any("pixel" in e and "changed" in e for e in v.qa.critical_errors)


def test_benchmark_runner_scene_compose_mock(tmp_path):
    import json as _j
    from PIL import Image as _I
    from renderello.benchmark import run_benchmark
    from renderello.webui import _mock_scene_fn
    src = tmp_path / "s.png"
    im = _I.new("RGB", (300, 400), (200, 200, 200))
    from PIL import ImageDraw
    ImageDraw.Draw(im).ellipse((90, 120, 210, 300), fill=(180, 40, 40))
    im.save(src)
    man = tmp_path / "m.json"
    man.write_text(_j.dumps({"routes": ["scene_compose"], "templates": ["clean_studio"],
                             "sources": [{"path": "s.png", "label": "t", "license": "test"}]}))
    cfg = RunConfig(out_root=tmp_path / "r", variants=1)
    csv_path = run_benchmark(man, cfg, scene_fn=_mock_scene_fn, scene_kind="mock")
    rep = _j.loads((csv_path.parent / "benchmark_report.json").read_text())
    assert "MOCK" in rep["scene_kind"]
    assert "mock" in csv_path.read_text()


def test_same_second_runs_never_reuse_success_artifacts(product_path, tmp_path, monkeypatch):
    """A failed retry must not inherit the first attempt's PNG or overwrite it."""
    monkeypatch.setattr("renderello.pipeline.time.strftime", lambda *_: "20261109-205000")
    cfg = RunConfig(seed=0, variants=1, out_root=tmp_path / "runs")
    req = EditRequest(source_path=product_path, template=get_template("clean_studio"))
    first = run_pipeline(req, [Route.PURE_PROMPT], cfg, editor=StubEditor())
    original_manifest = first.manifest_path.read_bytes()
    original_output = first.variants[0].output_path.read_bytes()

    class FailedEditor(StubEditor):
        def edit(self, *args, **kwargs):
            raise RuntimeError("test failure after earlier success")

    second = run_pipeline(req, [Route.PURE_PROMPT], cfg, editor=FailedEditor())
    assert first.run_dir != second.run_dir
    assert not second.variants[0].output_path.exists()
    assert json.loads(second.manifest_path.read_text())["variants"][0]["success"] is False
    assert first.manifest_path.read_bytes() == original_manifest
    assert first.variants[0].output_path.read_bytes() == original_output


def test_concurrent_run_directory_creation_is_unique(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from renderello.pipeline import _run_dir
    req = EditRequest(source_path=tmp_path / "same.png", template=get_template("clean_studio"))
    cfg = RunConfig(out_root=tmp_path / "runs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        dirs = list(pool.map(lambda _: _run_dir(cfg, req), range(32)))
    assert len(set(dirs)) == 32
    assert all(d.is_dir() for d in dirs)
